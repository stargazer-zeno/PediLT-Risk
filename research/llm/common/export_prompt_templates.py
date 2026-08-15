"""Export the exact SFT prompt components as standalone files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from llm.sft.build_sft_dataset import SYSTEM_PROMPT, USER_PREFIX_TEMPLATE, USER_SUFFIX_TEMPLATE


OUTPUT_SCHEMA = {"1m": 0.12, "1y": 0.26, "5y": 0.41}


def write_templates(directory: Path, purpose: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "system.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    (directory / "user_prefix.txt").write_text(USER_PREFIX_TEMPLATE + "\n", encoding="utf-8")
    (directory / "user_suffix.txt").write_text(USER_SUFFIX_TEMPLATE + "\n", encoding="utf-8")
    (directory / "output_schema.json").write_text(
        json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "README.md").write_text(
        f"# {purpose} prompt\n\n"
        "These files were exported from `llm/sft/build_sft_dataset.py`, which "
        "contains the final probability-only SFT prompt. The final user message "
        "is `user_prefix + formatted EHR + user_suffix`.\n",
        encoding="utf-8",
    )


def main() -> None:
    write_templates(RESEARCH_ROOT / "prompts" / "sft", "SFT")
    write_templates(RESEARCH_ROOT / "prompts" / "inference", "Inference")
    write_templates(RESEARCH_ROOT / "prompts" / "grpo", "GRPO")


if __name__ == "__main__":
    main()
