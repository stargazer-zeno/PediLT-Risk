"""Convert the private SFT JSONL into GRPO prompts with observed labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGETS = ("1m", "1y", "5y")


def valid_label(value: object) -> int | None:
    if value in (0, 1, True, False):
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    return None


def load_labels(gold_json: Path) -> dict[str, dict[str, int | None]]:
    items = json.loads(gold_json.read_text(encoding="utf-8"))
    return {
        str(item.get("id", "")): {
            target: valid_label((item.get("真实标签") or {}).get(target)) for target in TARGETS
        }
        for item in items
        if item.get("id")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create private GRPO prompts from SFT messages and observed labels.")
    parser.add_argument("--sft-jsonl", required=True, type=Path)
    parser.add_argument("--gold-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    args = parser.parse_args()

    labels_by_id = load_labels(args.gold_json)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_skipped = 0
    with args.sft_jsonl.open("r", encoding="utf-8") as source, args.output_jsonl.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            node_id = str(record.get("id", ""))
            labels = labels_by_id.get(node_id)
            if not node_id or labels is None or not any(value is not None for value in labels.values()):
                n_skipped += 1
                continue
            messages = record.get("messages") or []
            prompt = [message for message in messages if message.get("role") in {"system", "user"}]
            if len(prompt) != 2:
                n_skipped += 1
                continue
            target.write(json.dumps({"prompt": prompt, "true_labels": labels}, ensure_ascii=False) + "\n")
            n_written += 1
    print(json.dumps({"written": n_written, "skipped": n_skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
