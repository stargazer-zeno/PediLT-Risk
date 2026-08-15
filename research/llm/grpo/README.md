# GRPO training

This directory provides the complete probability-only GRPO training workflow.

1. Build a private prompt/label dataset:

```bash
python llm/grpo/prepare_grpo_dataset.py \
  --sft-jsonl artifacts/sft/xgb_sft_train.jsonl \
  --gold-json "$PEDILT_DATA_DIR/train_dataset_gold.json" \
  --output-jsonl artifacts/grpo/grpo_train.jsonl
```

2. Continue from a merged SFT checkpoint:

```bash
python llm/grpo/train_grpo.py \
  --model /path/to/merged-qwen3-4b-sft \
  --train-data artifacts/grpo/grpo_train.jsonl
```

The reward is `1 + 2*(1 - mean Brier error)` for strictly valid probability
JSON and `-1` for an invalid output. Labels marked `null` are excluded from the
Brier calculation. The script uses TRL's official GRPO implementation.
