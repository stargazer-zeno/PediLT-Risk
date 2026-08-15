"""Run probability-only inference against an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from llm.common.response_parser import parse_probability_json
from llm.sft.build_sft_dataset import SYSTEM_PROMPT, build_final_prompt


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen probability-only inference.")
    parser.add_argument("--input-json", required=True, type=Path, help="One raw patient JSON object or a JSON list.")
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--api-base", default=os.environ.get("LLM_API_BASE"), help="For example: http://HOST:PORT/v1")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "Qwen3-4B"))
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if not args.api_base:
        parser.error("--api-base or LLM_API_BASE is required")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install `openai` before running inference.") from exc

    client = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=900)
    records = load_records(args.input_json)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            user_prompt = build_final_prompt(record)
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw_output = completion.choices[0].message.content or ""
            parsed = parse_probability_json(raw_output)
            handle.write(
                json.dumps(
                    {
                        "record_index": index,
                        "source_id": record.get("id", ""),
                        "pred_probs": parsed,
                        "process_status": "success" if parsed is not None else "parse_failed",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
