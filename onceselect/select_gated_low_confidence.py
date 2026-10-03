from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fusion_common import GatedFusionSelector


def allocate_quotas(counts: np.ndarray, target: int) -> np.ndarray:
    raw = counts.astype(np.float64) * (target / int(counts.sum()))
    quotas = np.floor(raw).astype(np.int64)
    quotas[(counts > 0) & (quotas == 0)] = 1
    while int(quotas.sum()) < target:
        capacity = counts - quotas
        candidates = np.flatnonzero(capacity > 0)
        order = candidates[np.argsort(-(raw[candidates] - quotas[candidates]), kind="stable")]
        quotas[order[: min(target - int(quotas.sum()), len(order))]] += 1
    while int(quotas.sum()) > target:
        candidates = np.flatnonzero(quotas > 1)
        order = candidates[np.argsort(raw[candidates] - quotas[candidates], kind="stable")]
        quotas[order[: min(int(quotas.sum()) - target, len(order))]] -= 1
    return quotas


def build_model(checkpoint: dict, state_key: str, device: str):
    config = checkpoint["config"]
    model = GatedFusionSelector(
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        num_clusters=config["num_clusters"],
        dropout=config["dropout"],
        gate_hidden_dim=config["gate_hidden_dim"],
        fusion_mode=config.get("fusion_mode", "full"),
    ).to(device)
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    return model


def score(model, loader, device: str):
    confidence_chunks = []
    entropy_chunks = []
    gate_chunks = []
    with torch.inference_mode():
        for image, text in tqdm(loader, desc="Gated selector scoring"):
            image = image.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            logits, gate = model.forward_with_gate(image, text)
            probabilities = torch.softmax(logits, dim=-1)
            confidence_chunks.append(probabilities.max(dim=-1).values.cpu())
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            entropy_chunks.append(entropy.cpu())
            gate_chunks.append(gate.cpu())
    return torch.cat(confidence_chunks), torch.cat(entropy_chunks), torch.cat(gate_chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--select-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-alignment-gate", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image = payload["image_features"].float().contiguous()
    text = payload["text_features"].float().contiguous()
    record_indices = payload["record_indices"].long()
    labels = checkpoint["cluster_labels"].long()
    if not torch.equal(record_indices, checkpoint["record_indices"].long()):
        raise ValueError("Checkpoint and feature record indices differ")
    loader = DataLoader(
        TensorDataset(image, text), batch_size=args.batch_size, shuffle=False
    )
    selected_model = build_model(checkpoint, "selector_state_dict", args.device)
    confidence, entropy, image_gate = score(selected_model, loader, args.device)
    del selected_model
    early_model = build_model(checkpoint, "early_state_dict", args.device)
    early_confidence, _, _ = score(early_model, loader, args.device)
    confidence_growth = confidence - early_confidence

    own_similarity = (image * text).sum(dim=-1)
    cluster_text_mean = torch.zeros(
        checkpoint["config"]["num_clusters"], text.shape[1], dtype=torch.float32
    )
    cluster_counts = torch.bincount(
        labels, minlength=checkpoint["config"]["num_clusters"]
    ).float()
    cluster_text_mean.index_add_(0, labels, text)
    cluster_text_mean /= cluster_counts.clamp_min(1).unsqueeze(1)
    expected_other_similarity = (image * cluster_text_mean[labels]).sum(dim=-1)
    alignment_margin = own_similarity - expected_other_similarity
    detected_persistent_misalignment = (alignment_margin < 0) & (confidence_growth <= 0)
    persistent_misalignment = (
        torch.zeros_like(detected_persistent_misalignment)
        if args.disable_alignment_gate
        else detected_persistent_misalignment
    )

    counts = torch.bincount(labels, minlength=checkpoint["config"]["num_clusters"]).numpy()
    target = math.ceil(args.select_ratio * len(labels))
    quotas = allocate_quotas(counts, target)
    selected_rows: list[torch.Tensor] = []
    per_cluster: dict[str, dict] = {}
    fallback_total = 0
    for cluster_id, quota in enumerate(quotas.tolist()):
        members = torch.where(labels == cluster_id)[0]
        clean = members[~persistent_misalignment[members]]
        fallback = torch.empty(0, dtype=torch.long)
        if len(clean) < quota:
            rejected = members[persistent_misalignment[members]]
            fallback_count = quota - len(clean)
            fallback = rejected[
                torch.argsort(alignment_margin[rejected], descending=True, stable=True)[
                    :fallback_count
                ]
            ]
            fallback_total += len(fallback)
        clean_order = torch.argsort(confidence[clean], descending=False, stable=True)
        chosen = torch.cat((clean[clean_order[: min(quota, len(clean))]], fallback))
        selected_rows.append(chosen)
        per_cluster[str(cluster_id)] = {
            "candidates": int(len(members)),
            "persistent_misaligned_filtered": int(persistent_misalignment[members].sum()),
            "clean_candidates": int(len(clean)),
            "selected": int(len(chosen)),
            "fallback_misaligned": int(len(fallback)),
            "confidence_max_selected": float(confidence[chosen].max()),
            "alignment_margin_median": float(alignment_margin[members].median()),
        }
    selected_feature_rows = torch.cat(selected_rows)
    if len(selected_feature_rows) != target:
        raise RuntimeError(f"Selected {len(selected_feature_rows)}, expected {target}")
    selected_record_indices = sorted(record_indices[selected_feature_rows].tolist())
    records = json.loads(args.input_json.read_text(encoding="utf-8"))
    selected_records = [records[index] for index in selected_record_indices]
    prefix_counts = Counter(str(row.get("image", "")).split("/", 1)[0] for row in selected_records)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(selected_records, ensure_ascii=False))
    torch.save(
        {
            "record_indices": record_indices,
            "cluster_labels": labels,
            "confidence": confidence,
            "early_confidence": early_confidence,
            "confidence_growth": confidence_growth,
            "entropy": entropy,
            "image_gate": image_gate,
            "own_similarity": own_similarity,
            "expected_other_similarity": expected_other_similarity,
            "alignment_margin": alignment_margin,
            "persistent_misalignment": persistent_misalignment,
            "selected_feature_rows": selected_feature_rows,
        },
        args.scores,
    )
    levels = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    report = {
        "method": (
            "learnable gated image-text fusion; explicit elementwise interaction; "
            + (
                "no alignment filtering; "
                if args.disable_alignment_gate
                else "persistent relative-alignment noise gate; "
            )
            + "within-cluster lowest-confidence selection"
        ),
        "scored_multimodal_records": len(labels),
        "selected_records": len(selected_records),
        "select_ratio": args.select_ratio,
        "num_clusters": checkpoint["config"]["num_clusters"],
        "selector_epoch": checkpoint["selected_epoch"],
        "persistent_misaligned_filtered": int(persistent_misalignment.sum()),
        "persistent_misaligned_ratio": float(persistent_misalignment.float().mean()),
        "persistent_misaligned_detected": int(detected_persistent_misalignment.sum()),
        "alignment_gate_enabled": not args.disable_alignment_gate,
        "fallback_misaligned": fallback_total,
        "image_gate_quantiles": torch.quantile(image_gate, levels).tolist(),
        "alignment_margin_quantiles": torch.quantile(alignment_margin, levels).tolist(),
        "confidence_growth_quantiles": torch.quantile(confidence_growth, levels).tolist(),
        "selected_prefix_counts": dict(sorted(prefix_counts.items())),
        "per_cluster": per_cluster,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_cluster"}, indent=2))


if __name__ == "__main__":
    main()
