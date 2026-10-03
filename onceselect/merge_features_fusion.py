from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    indices = torch.cat([payload["record_indices"] for payload in payloads])
    image = torch.cat([payload["image_features"] for payload in payloads])
    text = torch.cat([payload["text_features"] for payload in payloads])
    order = torch.argsort(indices)
    indices, image, text = indices[order], image[order], text[order]
    if len(torch.unique(indices)) != len(indices):
        raise ValueError("Duplicate record indices across shards")
    metadata = dict(payloads[0]["metadata"])
    metadata.update(
        {
            "shard_index": None,
            "num_shards": len(payloads),
            "valid_multimodal": len(indices),
            "merged_inputs": [str(path.resolve()) for path in args.inputs],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_features": image,
            "text_features": text,
            "record_indices": indices,
            "metadata": metadata,
        },
        args.output,
    )
    print(f"saved={args.output} records={len(indices)} dim={image.shape[1]}")


if __name__ == "__main__":
    main()
