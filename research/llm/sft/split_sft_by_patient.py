"""Create a patient-disjoint SFT train/validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def patient_id(node_id: object) -> str:
    text = str(node_id)
    return text.split("_node_", 1)[0] if "_node_" in text else text


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("id"):
                raise ValueError(f"Missing id at {path}:{line_no}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-disjoint split for SFT JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--valid-output", required=True, type=Path)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be between 0 and 1")

    records = load_jsonl(args.input)
    groups = np.asarray([patient_id(record["id"]) for record in records])
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.valid_ratio, random_state=args.seed)
    train_idx, valid_idx = next(splitter.split(np.zeros(len(records)), groups=groups))
    train_records = [records[idx] for idx in train_idx]
    valid_records = [records[idx] for idx in valid_idx]
    train_patients = {patient_id(record["id"]) for record in train_records}
    valid_patients = {patient_id(record["id"]) for record in valid_records}
    if train_patients & valid_patients:
        raise RuntimeError("Patient overlap detected after split")

    write_jsonl(args.train_output, train_records)
    write_jsonl(args.valid_output, valid_records)
    print(
        json.dumps(
            {
                "train_nodes": len(train_records),
                "valid_nodes": len(valid_records),
                "train_patients": len(train_patients),
                "valid_patients": len(valid_patients),
                "patient_overlap": 0,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
