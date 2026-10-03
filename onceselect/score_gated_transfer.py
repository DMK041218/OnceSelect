#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from fusion_common import GatedFusionSelector


def build_model(checkpoint: dict, device: str) -> GatedFusionSelector:
    config = checkpoint["config"]
    model = GatedFusionSelector(
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        num_clusters=config["num_clusters"],
        dropout=config["dropout"],
        gate_hidden_dim=config["gate_hidden_dim"],
        fusion_mode=config.get("fusion_mode", "full"),
    ).to(device)
    model.load_state_dict(checkpoint["selector_state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-selector", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.source_selector, map_location="cpu", weights_only=False)
    image = payload["image_features"]
    text = payload["text_features"]
    record_indices = payload["record_indices"].long()
    if image.shape != text.shape:
        raise ValueError("Image and text features must have identical shapes")
    if image.shape[1] != checkpoint["config"]["feature_dim"]:
        raise ValueError("Target feature dimension does not match source selector")

    centers = checkpoint["cluster_centers"].float().to(args.device)
    center_norm = centers.square().sum(dim=1).unsqueeze(0)
    model = build_model(checkpoint, args.device)
    confidence_chunks = []
    entropy_chunks = []
    gate_chunks = []
    label_chunks = []
    for start in tqdm(range(0, len(record_indices), args.batch_size), desc="VF transfer scoring"):
        stop = min(start + args.batch_size, len(record_indices))
        batch_image = image[start:stop].to(args.device, dtype=torch.float32, non_blocking=True)
        batch_text = text[start:stop].to(args.device, dtype=torch.float32, non_blocking=True)
        with torch.inference_mode():
            logits, gate = model.forward_with_gate(batch_image, batch_text)
            probabilities = torch.softmax(logits, dim=-1)
            confidence = probabilities.max(dim=-1).values
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            joint = F.normalize(torch.cat((batch_image, batch_text), dim=-1), dim=-1)
            distance = (
                joint.square().sum(dim=1, keepdim=True)
                + center_norm
                - 2 * joint @ centers.T
            )
            labels = distance.argmin(dim=-1)
        confidence_chunks.append(confidence.cpu())
        entropy_chunks.append(entropy.cpu())
        gate_chunks.append(gate.cpu())
        label_chunks.append(labels.cpu())

    confidence = torch.cat(confidence_chunks)
    entropy = torch.cat(entropy_chunks)
    image_gate = torch.cat(gate_chunks)
    labels = torch.cat(label_chunks)
    num_clusters = checkpoint["config"]["num_clusters"]
    histogram = torch.bincount(labels, minlength=num_clusters)
    levels = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    score_payload = {
        "record_indices": record_indices,
        "cluster_labels": labels,
        "confidence": confidence,
        "entropy": entropy,
        "image_gate": image_gate,
        "metadata": {
            "method": "frozen gated selector transfer with nearest source KMeans center labels",
            "source_selector": str(args.source_selector.resolve()),
            "target_features": str(args.features.resolve()),
            "source_selector_epoch": checkpoint["selected_epoch"],
            "source_selector_macro_accuracy": checkpoint["selected_validation"]["macro_accuracy"],
        },
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    torch.save(score_payload, args.scores)
    report = {
        **score_payload["metadata"],
        "records_scored": len(record_indices),
        "num_clusters": num_clusters,
        "assigned_cluster_coverage": int((histogram > 0).sum()),
        "assigned_cluster_histogram": histogram.tolist(),
        "confidence_quantiles": torch.quantile(confidence, levels).tolist(),
        "entropy_quantiles": torch.quantile(entropy, levels).tolist(),
        "image_gate_quantiles": torch.quantile(image_gate, levels).tolist(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "assigned_cluster_histogram"}, indent=2))


if __name__ == "__main__":
    main()
