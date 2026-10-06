[Uploading 10.1007_978-3-032-37577-3_29-citation (1).bib…]()
# OnceSelect

Official implementation and pretrained artifacts for **OnceSelect: a reusable data selector for multimodal instruction tuning**.

OnceSelect learns a semantic reference once from a source instruction-tuning dataset, then reuses the frozen selector to score unseen candidate datasets. It combines cluster-wise low-confidence candidate construction with uncertainty-weighted, blockwise fixed-cardinality DPP sampling.

## Released artifacts

This repository intentionally contains only the paper-default implementation and lightweight metadata:

- `checkpoints/selector_b32/selector_training.json`: training history and checkpoint-selection metadata.
- `onceselect/`: the core feature extraction, selector training, transfer scoring, and k-DPP selection code.
- The default `selector.pt`, its `clustering.pt`, and the best LLaVA-v1.5-7B LoRA trained on the CLIP-B/32-selected subset are hosted in the [OnceSelect Hugging Face repository](https://huggingface.co/aaakiyasuqqqa/OnceSelect).

Raw datasets, extracted features, evaluation datasets, and alternative backbone/ablation checkpoints are not included.

## Default selector

The released selector uses:

| Setting | Value |
|---|---:|
| Frozen encoder | CLIP ViT-B/32 |
| Feature dimension per modality | 512 |
| Semantic clusters | 128 |
| Centroid-nearest core | 3% per cluster |
| Validation split | 20% of each cluster core |
| Selector hidden dimension | 256 |
| Gate hidden dimension | 128 |
| Learning rate | `1e-5` |
| Selector batch size | 1024 |
| Target validation macro accuracy | 0.60 |
| Selected epoch | 35 |
| Actual validation macro accuracy | 0.61078 |
| Seed | 42 |

The architecture contains separate image and text projection branches, a sample-dependent scalar gate, and an elementwise image-text interaction branch.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download a CLIP ViT-B/32 checkpoint supported by Hugging Face Transformers, or point `--model` to an existing local copy. Feature extraction uses local model files by default.

## Select from a new dataset

Input annotations are expected to be a JSON list. Each record should contain an `image` path and either LLaVA-style `conversations` or an instruction-like text field. The following example uses one feature shard; use multiple shard indices in parallel for large datasets.

### 1. Extract and merge frozen multimodal features

```bash
python onceselect/extract_features_fusion.py \
  --input-json /path/to/data.json \
  --image-root /path/to/images \
  --model /path/to/clip-vit-base-patch32 \
  --output work/features/shard_00_of_1.pt \
  --batch-size 128 --num-workers 4 \
  --device cuda:0 --shard-index 0 --num-shards 1

python onceselect/merge_features_fusion.py \
  --inputs work/features/shard_00_of_1.pt \
  --output work/features.pt
```

### 2. Reuse the frozen selector

The transfer scorer assigns each target sample to its nearest source centroid and records selector confidence.

```bash
python onceselect/score_gated_transfer.py \
  --features work/features.pt \
  --source-selector checkpoints/selector_b32/selector.pt \
  --scores work/scores.pt \
  --report work/scoring_report.json \
  --batch-size 8192 --device cuda:0
```

### 3. Apply cluster-wise uncertainty-weighted k-DPP

```bash
python onceselect/select_kdpp_low_confidence.py \
  --features work/features.pt \
  --scores work/scores.pt \
  --input-json /path/to/data.json \
  --output-json work/selected_15pct.json \
  --selection-state work/selection_state.pt \
  --report work/selection_report.json \
  --select-ratio 0.15 \
  --candidate-multiplier 2.0 \
  --max-block-size 128 \
  --seed 42
```

## Train the default selector from scratch

After extracting features from a source dataset, run:

```bash
python onceselect/train_gated_fusion_selector.py \
  --features work/source_features.pt \
  --clustering-checkpoint work/source_clustering.pt \
  --checkpoint work/selector.pt \
  --num-clusters 128 \
  --core-ratio 0.03 \
  --val-ratio 0.20 \
  --hidden-dim 256 \
  --gate-hidden-dim 128 \
  --fusion-mode full \
  --batch-size 1024 \
  --learning-rate 1e-5 \
  --target-macro-accuracy 0.60 \
  --checkpoint-policy target \
  --seed 42 --device cuda:0
```

## LLaVA-7B LoRA

The released LoRA is the best adapter trained on the subset selected by the default CLIP-B/32 selector. It uses:

- base language model: `lmsys/vicuna-7b-v1.5`;
- LLaVA-v1.5-7B multimodal projector;
- downstream vision tower: `openai/clip-vit-large-patch14-336`;
- LoRA rank 128, alpha 256, dropout 0.05;
- one training epoch with per-device batch size 16.

Download the LoRA from [`aaakiyasuqqqa/OnceSelect`](https://huggingface.co/aaakiyasuqqqa/OnceSelect/tree/main/llava7b_lora_b32) and verify it against the published checksum file before use.


## Citation

```
@misc{dong2026onceselectreusabledataselection,
      title={OnceSelect: Reusable Data Selection for Efficient Multimodal Instruction Tuning}, 
      author={Mingkang Dong and Muxin Pu and Hongyi Cai and JieLi and Jiancheng Pan and Xu Zheng and Yadan Luo and Yuqian Fu},
      year={2026},
      eprint={2605.26761},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.26761}, 
}
@misc{dong2026visnecmeasuringleveragingvisual,
      title={VisNec: Measuring and Leveraging Visual Necessity for Multimodal Instruction Tuning}, 
      author={Mingkang Dong and Hongyi Cai and Jie Li and Sifan Zhou and Bin Ren and Kunyu Peng and Yuqian Fu},
      year={2026},
      eprint={2603.01195},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.01195}, 
}
```
