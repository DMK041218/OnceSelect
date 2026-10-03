from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from fusion_common import GatedFusionSelector, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--clustering-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-clusters", type=int, default=20)
    parser.add_argument("--core-ratio", type=float, default=0.03)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--fusion-mode",
        choices=("full", "static_gate", "no_interaction"),
        default="full",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-macro-accuracy", type=float, default=0.85)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("target", "target_any", "last", "max_coverage"),
        default="target",
        help=(
            "Choose the checkpoint closest to target macro accuracy after full "
            "coverage (baseline), closest across any epoch, "
            "always use the final epoch, or use the earliest epoch that reaches "
            "the maximum observed cluster coverage."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(model, loader, device: str, num_clusters: int) -> dict:
    model.eval()
    correct = 0
    seen = 0
    loss_sum = 0.0
    class_correct = torch.zeros(num_clusters, dtype=torch.long)
    class_seen = torch.zeros(num_clusters, dtype=torch.long)
    predicted_histogram = torch.zeros(num_clusters, dtype=torch.long)
    confidence_chunks: list[torch.Tensor] = []
    gate_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for image, text, labels in loader:
            image = image.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, gate = model.forward_with_gate(image, text)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            probabilities = torch.softmax(logits, dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            predicted_histogram += torch.bincount(prediction.cpu(), minlength=num_clusters)
            confidence_chunks.append(confidence.cpu())
            gate_chunks.append(gate.cpu())
            correct += (prediction == labels).sum().item()
            seen += labels.numel()
            for cluster_id in range(num_clusters):
                mask = labels == cluster_id
                class_seen[cluster_id] += mask.sum().cpu()
                class_correct[cluster_id] += (prediction[mask] == cluster_id).sum().cpu()
    per_class = class_correct.float() / class_seen.clamp_min(1)
    confidence = torch.cat(confidence_chunks)
    gate = torch.cat(gate_chunks)
    levels = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        "loss": loss_sum / seen,
        "accuracy": correct / seen,
        "macro_accuracy": per_class.mean().item(),
        "per_class_accuracy": per_class.tolist(),
        "predicted_cluster_coverage": int((predicted_histogram > 0).sum()),
        "predicted_cluster_histogram": predicted_histogram.tolist(),
        "confidence_quantiles": torch.quantile(confidence, levels).tolist(),
        "image_gate_quantiles": torch.quantile(gate, levels).tolist(),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    image = payload["image_features"].float().contiguous()
    text = payload["text_features"].float().contiguous()
    record_indices = payload["record_indices"].long()
    if image.shape != text.shape:
        raise ValueError("Image and text features must have equal shapes")

    if args.clustering_checkpoint.is_file():
        cluster_payload = torch.load(
            args.clustering_checkpoint, map_location="cpu", weights_only=False
        )
        if not torch.equal(record_indices, cluster_payload["record_indices"].long()):
            raise ValueError("Clustering record indices do not match features")
        labels_np = cluster_payload["labels"].numpy().astype(np.int64)
        centers_np = cluster_payload["centers"].numpy().astype(np.float32)
        print(f"reused_clustering={args.clustering_checkpoint}", flush=True)
    else:
        joint = torch.cat((image, text), dim=-1)
        joint = F.normalize(joint, dim=-1).numpy()
        kmeans = MiniBatchKMeans(
            n_clusters=args.num_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            batch_size=8192,
            max_no_improvement=30,
            reassignment_ratio=0.01,
            random_state=args.seed,
            verbose=1,
        )
        labels_np = kmeans.fit_predict(joint).astype(np.int64)
        centers_np = kmeans.cluster_centers_.astype(np.float32)
        args.clustering_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "labels": torch.from_numpy(labels_np),
                "centers": torch.from_numpy(centers_np),
                "record_indices": record_indices,
                "config": {
                    "num_clusters": args.num_clusters,
                    "feature": (
                        "normalized concat(dual-encoder image, mean-turn text): "
                        f"{payload.get('metadata', {}).get('clip_model', 'unknown')}"
                    ),
                    "seed": args.seed,
                },
            },
            args.clustering_checkpoint,
        )
        del joint

    joint = F.normalize(torch.cat((image, text), dim=-1), dim=-1).numpy()
    rng = np.random.default_rng(args.seed)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    cluster_sizes: dict[str, int] = {}
    core_sizes: dict[str, int] = {}
    for cluster_id in range(args.num_clusters):
        members = np.flatnonzero(labels_np == cluster_id)
        distances = np.linalg.norm(joint[members] - centers_np[cluster_id], axis=1)
        core_count = max(2, math.ceil(args.core_ratio * len(members)))
        core = members[np.argsort(distances, kind="stable")[:core_count]].copy()
        rng.shuffle(core)
        val_count = max(1, round(args.val_ratio * len(core)))
        val_indices.append(core[:val_count])
        train_indices.append(core[val_count:])
        cluster_sizes[str(cluster_id)] = int(len(members))
        core_sizes[str(cluster_id)] = int(core_count)
    del joint

    train_np = np.concatenate(train_indices)
    val_np = np.concatenate(val_indices)
    labels = torch.from_numpy(labels_np)
    train_labels = labels[torch.from_numpy(train_np)]
    class_counts = torch.bincount(train_labels, minlength=args.num_clusters).float()
    weights = (1.0 / class_counts.clamp_min(1))[train_labels]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_np),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_rows = torch.from_numpy(train_np)
    val_rows = torch.from_numpy(val_np)
    train_loader = DataLoader(
        TensorDataset(image[train_rows], text[train_rows], labels[train_rows]),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(image[val_rows], text[val_rows], labels[val_rows]),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = GatedFusionSelector(
        feature_dim=image.shape[1],
        hidden_dim=args.hidden_dim,
        num_clusters=args.num_clusters,
        dropout=args.dropout,
        gate_hidden_dim=args.gate_hidden_dim,
        fusion_mode=args.fusion_mode,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict] = []
    states: list[dict[str, torch.Tensor]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for batch_image, batch_text, batch_labels in tqdm(
            train_loader, desc=f"Gated selector epoch {epoch}"
        ):
            batch_image = batch_image.to(args.device, non_blocking=True)
            batch_text = batch_text.to(args.device, non_blocking=True)
            batch_labels = batch_labels.to(args.device, non_blocking=True)
            logits = model(batch_image, batch_text)
            loss = F.cross_entropy(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch_labels.numel()
            correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
            seen += batch_labels.numel()
        validation = evaluate(model, val_loader, args.device, args.num_clusters)
        metrics = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            **validation,
        }
        history.append(metrics)
        states.append(copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
        print(json.dumps(metrics), flush=True)

    if args.checkpoint_policy == "last":
        selected_index = len(history) - 1
    elif args.checkpoint_policy == "target_any":
        selected_index = min(
            range(len(history)),
            key=lambda index: (
                abs(history[index]["macro_accuracy"] - args.target_macro_accuracy),
                history[index]["epoch"],
            ),
        )
    elif args.checkpoint_policy == "max_coverage":
        maximum_coverage = max(
            metrics["predicted_cluster_coverage"] for metrics in history
        )
        selected_index = next(
            index
            for index, metrics in enumerate(history)
            if metrics["predicted_cluster_coverage"] == maximum_coverage
        )
    else:
        complete = [
            index
            for index, metrics in enumerate(history)
            if metrics["predicted_cluster_coverage"] == args.num_clusters
        ]
        pool = complete if complete else list(range(len(history)))
        selected_index = min(
            pool,
            key=lambda index: (
                abs(history[index]["macro_accuracy"] - args.target_macro_accuracy),
                history[index]["epoch"],
            ),
        )
    config = {
        "feature_dim": int(image.shape[1]),
        "hidden_dim": args.hidden_dim,
        "gate_hidden_dim": args.gate_hidden_dim,
        "fusion_mode": args.fusion_mode,
        "num_clusters": args.num_clusters,
        "dropout": args.dropout,
        "core_ratio": args.core_ratio,
        "val_ratio": args.val_ratio,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "target_macro_accuracy": args.target_macro_accuracy,
        "checkpoint_policy": args.checkpoint_policy,
        "seed": args.seed,
        "architecture": {
            "full": "dual branch scalar gate plus elementwise CLIP interaction",
            "static_gate": "dual branch fixed 0.5 gate plus elementwise CLIP interaction",
            "no_interaction": "dual branch scalar gate without interaction residual branch",
        }[args.fusion_mode],
    }
    checkpoint = {
        "selector_state_dict": states[selected_index],
        "early_state_dict": states[0],
        "selected_epoch": history[selected_index]["epoch"],
        "selected_validation": history[selected_index],
        "early_validation": history[0],
        "history": history,
        "config": config,
        "cluster_labels": torch.from_numpy(labels_np),
        "cluster_centers": torch.from_numpy(centers_np),
        "record_indices": record_indices,
        "cluster_sizes": cluster_sizes,
        "core_sizes": core_sizes,
        "feature_metadata": payload["metadata"],
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    args.checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": config,
                "selected_epoch": history[selected_index]["epoch"],
                "selected_validation": history[selected_index],
                "early_validation": history[0],
                "history": history,
                "cluster_sizes": cluster_sizes,
                "core_sizes": core_sizes,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "selected_epoch": history[selected_index]["epoch"],
                "selected_macro_accuracy": history[selected_index]["macro_accuracy"],
                "coverage": history[selected_index]["predicted_cluster_coverage"],
            }
        )
    )


if __name__ == "__main__":
    main()
