from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from kdpp_module import kdpp_sampling


def allocate_quotas(counts: np.ndarray, target: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    if target < 0 or target > int(counts.sum()):
        raise ValueError("target is outside available capacity")
    if target == 0:
        return np.zeros_like(counts)
    raw = counts.astype(np.float64) * (target / int(counts.sum()))
    quotas = np.floor(raw).astype(np.int64)
    if target >= int((counts > 0).sum()):
        quotas[(counts > 0) & (quotas == 0)] = 1
    while int(quotas.sum()) < target:
        candidates = np.flatnonzero(quotas < counts)
        order = candidates[
            np.argsort(-(raw[candidates] - quotas[candidates]), kind="stable")
        ]
        take = min(target - int(quotas.sum()), len(order))
        quotas[order[:take]] += 1
    while int(quotas.sum()) > target:
        candidates = np.flatnonzero(quotas > 0)
        order = candidates[
            np.argsort(raw[candidates] - quotas[candidates], kind="stable")
        ]
        take = min(int(quotas.sum()) - target, len(order))
        quotas[order[:take]] -= 1
    return quotas


def rank_quality(confidence: np.ndarray) -> np.ndarray:
    """Map lower confidence to larger, scale-free quality in (0, 1]."""
    order = np.argsort(confidence, kind="stable")
    quality = np.empty(len(order), dtype=np.float64)
    quality[order] = 1.0 - np.arange(len(order), dtype=np.float64) / len(order)
    return quality


def blockwise_kdpp(
    features: np.ndarray,
    quality: np.ndarray,
    target: int,
    *,
    max_block_size: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Approximate a large k-DPP by exact k-DPPs on deterministic blocks."""
    n = len(features)
    if target == n:
        return np.arange(n, dtype=np.int64), 1
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    blocks = [permutation[start : start + max_block_size] for start in range(0, n, max_block_size)]
    block_counts = np.asarray([len(block) for block in blocks], dtype=np.int64)
    block_quotas = allocate_quotas(block_counts, target)
    chosen: list[np.ndarray] = []
    for block_id, (block, quota) in enumerate(zip(blocks, block_quotas.tolist())):
        if quota == 0:
            continue
        local = kdpp_sampling(
            features[block],
            quality[block],
            quota,
            seed=seed + block_id + 1,
        )
        chosen.append(block[local])
    result = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    if len(result) != target or len(np.unique(result)) != target:
        raise RuntimeError(f"blockwise k-DPP selected {len(result)}, expected {target}")
    return result, len(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--selection-state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--select-ratio", type=float, default=0.15)
    parser.add_argument(
        "--selection-count",
        type=int,
        default=None,
        help="Select an exact number of records instead of deriving it from select-ratio.",
    )
    parser.add_argument("--candidate-multiplier", type=float, default=2.0)
    parser.add_argument("--max-block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.select_ratio <= 1:
        raise ValueError("select-ratio must be in (0, 1]")
    if args.selection_count is not None and args.selection_count <= 0:
        raise ValueError("selection-count must be positive")
    if args.candidate_multiplier < 1:
        raise ValueError("candidate-multiplier must be >= 1")
    if args.max_block_size < 2:
        raise ValueError("max-block-size must be >= 2")

    feature_payload = torch.load(args.features, map_location="cpu", weights_only=False)
    score_payload = torch.load(args.scores, map_location="cpu", weights_only=False)
    image = feature_payload["image_features"].float()
    text = feature_payload["text_features"].float()
    record_indices = feature_payload["record_indices"].long()
    confidence = score_payload["confidence"].float().numpy()
    labels = score_payload["cluster_labels"].long().numpy()
    if not torch.equal(record_indices, score_payload["record_indices"].long()):
        raise ValueError("feature and score record indices differ")
    if len(image) != len(confidence) or len(image) != len(labels):
        raise ValueError("feature and score lengths differ")

    # The DPP sees the same normalized joint CLIP geometry used by KMeans.
    joint = F.normalize(torch.cat((image, text), dim=-1), dim=-1).numpy()
    num_clusters = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_clusters)
    target = (
        args.selection_count
        if args.selection_count is not None
        else math.ceil(args.select_ratio * len(labels))
    )
    if target > len(labels):
        raise ValueError("selection-count exceeds the number of scored records")
    quotas = allocate_quotas(counts, target)

    selected_rows: list[np.ndarray] = []
    report_clusters: dict[str, dict] = {}
    total_blocks = 0
    for cluster_id, quota in enumerate(tqdm(quotas.tolist(), desc="cluster k-DPP")):
        members = np.flatnonzero(labels == cluster_id)
        if quota == 0:
            report_clusters[str(cluster_id)] = {
                "members": int(len(members)),
                "candidate_count": 0,
                "selected": 0,
                "blocks": 0,
                "selected_confidence_mean": None,
                "selected_confidence_max": None,
            }
            continue
        order = np.argsort(confidence[members], kind="stable")
        candidate_count = min(
            len(members), max(quota, math.ceil(args.candidate_multiplier * quota))
        )
        candidates = members[order[:candidate_count]]
        quality = rank_quality(confidence[candidates])
        local_rows, block_count = blockwise_kdpp(
            joint[candidates],
            quality,
            quota,
            max_block_size=args.max_block_size,
            seed=args.seed + cluster_id * 100_000,
        )
        chosen = candidates[local_rows]
        selected_rows.append(chosen)
        total_blocks += block_count
        report_clusters[str(cluster_id)] = {
            "members": int(len(members)),
            "candidate_count": int(candidate_count),
            "selected": int(len(chosen)),
            "blocks": int(block_count),
            "selected_confidence_mean": float(confidence[chosen].mean()),
            "selected_confidence_max": float(confidence[chosen].max()),
        }

    selected_feature_rows = np.concatenate(selected_rows)
    if len(selected_feature_rows) != target or len(np.unique(selected_feature_rows)) != target:
        raise RuntimeError(f"selected {len(selected_feature_rows)}, expected {target}")
    chosen_record_indices = sorted(record_indices[selected_feature_rows].tolist())
    records = json.loads(args.input_json.read_text(encoding="utf-8"))
    selected_records = [records[index] for index in chosen_record_indices]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.selection_state.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(selected_records, ensure_ascii=False))
    torch.save(
        {
            "selected_feature_rows": torch.from_numpy(selected_feature_rows),
            "selected_record_indices": torch.tensor(chosen_record_indices),
            "config": vars(args),
        },
        args.selection_state,
    )
    report = {
        "method": "within-cluster low-confidence candidate pool followed by blockwise exact k-DPP",
        "global_exact_kdpp": False,
        "reason": "global N x N exact k-DPP is infeasible at LLaVA-665K scale",
        "records_scored": int(len(labels)),
        "selected_records": int(len(selected_records)),
        "select_ratio": target / len(labels),
        "requested_select_ratio": args.select_ratio,
        "selection_count": target,
        "candidate_multiplier": args.candidate_multiplier,
        "candidate_ratio_approx": target / len(labels) * args.candidate_multiplier,
        "max_block_size": args.max_block_size,
        "total_exact_kdpp_blocks": total_blocks,
        "kernel": "quality-weighted shifted-cosine L-ensemble with diagonal jitter",
        "seed": args.seed,
        "per_cluster": report_clusters,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_cluster"}, indent=2))


if __name__ == "__main__":
    main()
