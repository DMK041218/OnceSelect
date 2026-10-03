from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON list")
    return payload


def human_turns(record: dict) -> list[str]:
    turns: list[str] = []
    for turn in record.get("conversations", []):
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", turn.get("role", ""))).lower()
        value = turn.get("value", turn.get("content", ""))
        if role in {"human", "user"} and isinstance(value, str) and value.strip():
            turns.append(value.replace("<image>", " ").strip())
    if turns:
        return turns
    for key in ("instruction", "question", "query", "prompt", "text", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return [value.replace("<image>", " ").strip()]
    return []


def resolve_image(root: Path, reference: str) -> Path | None:
    ref = Path(reference)
    candidates = [ref] if ref.is_absolute() else []
    candidates.extend((root / ref, root / ref.name))
    return next((path for path in candidates if path.is_file()), None)


class FusionDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        image_root: Path,
        shard_index: int,
        num_shards: int,
    ) -> None:
        self.samples: list[tuple[int, str, list[str]]] = []
        missing_image = 0
        missing_text = 0
        for index, record in enumerate(tqdm(records, desc="Resolving samples")):
            if index % num_shards != shard_index:
                continue
            texts = human_turns(record)
            if not texts:
                missing_text += 1
                continue
            image_ref = record.get("image")
            if not isinstance(image_ref, str):
                missing_image += 1
                continue
            image_path = resolve_image(image_root, image_ref)
            if image_path is None:
                missing_image += 1
                continue
            self.samples.append((index, str(image_path), texts))
        self.stats = {
            "total_records": len(records),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "valid_multimodal": len(self.samples),
            "missing_or_unresolved_image": missing_image,
            "missing_instruction": missing_text,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        record_index, image_path, texts = self.samples[index]
        try:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
            return record_index, rgb, texts
        except Exception:
            return None


class FusionCollator:
    def __init__(self, processor, text_max_length: int) -> None:
        self.processor = processor
        self.text_max_length = text_max_length

    def __call__(self, batch):
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            return None
        indices, images, turn_lists = zip(*batch)
        image_inputs = self.processor(images=list(images), return_tensors="pt")
        flat_turns = [text for turns in turn_lists for text in turns]
        text_inputs = self.processor(
            text=flat_turns,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
        )
        turn_counts = torch.tensor([len(turns) for turns in turn_lists], dtype=torch.long)
        return (
            torch.tensor(indices, dtype=torch.long),
            dict(image_inputs),
            dict(text_inputs),
            turn_counts,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def projected_tensor(value):
    return value.pooler_output if hasattr(value, "pooler_output") else value


def main() -> None:
    args = parse_args()
    records = load_records(args.input_json)
    dataset = FusionDataset(records, args.image_root, args.shard_index, args.num_shards)
    if not dataset:
        raise RuntimeError("No valid samples in shard")
    print(json.dumps(dataset.stats, indent=2), flush=True)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    text_config = getattr(model.config, "text_config", model.config)
    text_max_length = int(getattr(text_config, "max_position_embeddings", 77))
    if not hasattr(model, "get_image_features") or not hasattr(model, "get_text_features"):
        raise TypeError(
            f"{type(model).__name__} is not a compatible dual encoder: "
            "get_image_features/get_text_features are required"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=FusionCollator(processor, text_max_length),
    )

    image_chunks: list[torch.Tensor] = []
    text_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Fusion features shard {args.shard_index}"):
            if batch is None:
                continue
            indices, image_inputs, text_inputs, turn_counts = batch
            image_inputs = {
                key: value.to(args.device, non_blocking=True)
                for key, value in image_inputs.items()
            }
            text_inputs = {
                key: value.to(args.device, non_blocking=True)
                for key, value in text_inputs.items()
            }
            image = projected_tensor(model.get_image_features(**image_inputs))
            flat_text = projected_tensor(model.get_text_features(**text_inputs))
            image = F.normalize(image, dim=-1)
            flat_text = F.normalize(flat_text, dim=-1)
            split_text = torch.split(flat_text, turn_counts.tolist())
            text = F.normalize(
                torch.stack([turn.mean(dim=0) for turn in split_text]), dim=-1
            )
            image_chunks.append(image.to(dtype=torch.float16, device="cpu"))
            text_chunks.append(text.to(dtype=torch.float16, device="cpu"))
            index_chunks.append(indices)

    payload = {
        "image_features": torch.cat(image_chunks),
        "text_features": torch.cat(text_chunks),
        "record_indices": torch.cat(index_chunks),
        "metadata": {
            **dataset.stats,
            "input_json": str(args.input_json.resolve()),
            "image_root": str(args.image_root.resolve()),
            "clip_model": str(Path(args.model).resolve()),
            "dual_encoder_class": type(model).__name__,
            "text_max_length": text_max_length,
            "feature_normalization": "separate L2 image/text; average normalized human-turn text; renormalize",
            "projection_dim": int(image_chunks[0].shape[1]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved={args.output} records={len(payload['record_indices'])}")


if __name__ == "__main__":
    main()
