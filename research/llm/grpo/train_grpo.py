"""Train a Qwen3-4B SFT checkpoint with GRPO probability rewards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from llm.grpo.reward import probability_grpo_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training from an SFT LoRA or merged checkpoint.")
    parser.add_argument("--model", required=True, help="Merged SFT checkpoint or base model with the SFT adapter applied.")
    parser.add_argument("--train-data", required=True, help="Private JSONL made by prepare_grpo_dataset.py")
    parser.add_argument("--output-dir", default="artifacts/checkpoints/qwen3-4b-grpo")
    parser.add_argument("--max-prompt-length", type=int, default=32768)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "GRPO dependencies are missing. Install the optional requirements with "
            "`pip install -r requirements-grpo.txt`."
        ) from exc

    dataset = load_dataset("json", data_files=args.train_data, split="train")
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        bf16=True,
        seed=42,
        temperature=0.9,
        num_generations=8,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=128,
        beta=0.04,
        epsilon=0.2,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=probability_grpo_reward,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
