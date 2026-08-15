#!/usr/bin/env bash
# Organized release launcher.  Derived from the final Qwen3-4B LoRA launcher.
# Inputs are deliberately supplied through environment variables; do not embed
# patient-data paths, model-cache paths, or credentials in this file.
set -euo pipefail

: "${QWEN_BASE_MODEL:=Qwen/Qwen3-4B}"
: "${SFT_TRAIN_DATA:=artifacts/sft/train.jsonl}"
: "${SFT_VALID_DATA:=artifacts/sft/valid.jsonl}"
: "${SFT_OUTPUT_DIR:=artifacts/checkpoints/qwen3-4b-sft}"

mkdir -p "${SFT_OUTPUT_DIR}"

swift sft \
  --model "${QWEN_BASE_MODEL}" \
  --train_type lora \
  --dataset "${SFT_TRAIN_DATA}" \
  --val_dataset "${SFT_VALID_DATA}" \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --max_length 32768 \
  --gradient_checkpointing true \
  --attn_impl flash_attn \
  --eval_steps 500 \
  --save_steps 500 \
  --save_total_limit 3 \
  --output_dir "${SFT_OUTPUT_DIR}"
