"""Summarize inference records that can be retried under a frozen protocol."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "evaluation" / "statistical_significance" / "results" / "statistical_audit"
XGB_SOURCE = PROJECT_ROOT / "machine_learning" / "train" / "xgboost" / "xgb_sequence_saved_test_predictions.csv"
STAGE_JSON = Path(os.environ.get("PEDILT_STAGE_MAP", PROJECT_ROOT / "private_inputs" / "test_ids_by_stage.json"))

TARGET_MAP = {"Label_1m": "1m", "Label_1y": "1y", "Label_5y": "5y"}


def load_reference() -> pd.DataFrame:
    rows = []
    with XGB_SOURCE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = TARGET_MAP.get(row["Target"])
            if target is None:
                continue
            rows.append(
                {
                    "Sample_ID": row["Sample_ID"],
                    "Target": target,
                    "True_Label": int(float(row["True_Label"])),
                }
            )
    return pd.DataFrame(rows)


def load_stage_map() -> dict[str, str]:
    data = json.loads(STAGE_JSON.read_text(encoding="utf-8"))
    stage_map = {}
    for stage, ids in data.items():
        normalized = stage[:1].upper() + stage[1:] if stage.startswith("stage") else stage
        for sample_id in ids:
            stage_map[str(sample_id)] = normalized
    return stage_map


def main() -> None:
    reference = load_reference()
    stage_map = load_stage_map()
    reference["Stage"] = reference["Sample_ID"].map(stage_map).fillna("Unknown")

    invalid = pd.read_csv(AUDIT_DIR / "llm_invalid_or_missing_target_records.csv")
    invalid = invalid.merge(reference, on=["Sample_ID", "Target"], how="left")
    invalid["True_Label"] = invalid["True_Label"].fillna(-1).astype(int)
    invalid["Stage"] = invalid["Stage"].fillna("Unknown")

    recoverable = invalid[
        invalid["Status"].isin(
            ["missing_record", "api_failed", "parse_failed", "missing_pred_prob", "invalid_pred_prob"]
        )
    ].copy()

    target_rows = []
    for (model, target), group in recoverable.groupby(["Model", "Target"], sort=False):
        target_rows.append(
            {
                "Model": model,
                "Target": target,
                "Recoverable_N": len(group),
                "Recoverable_Positive_N": int((group["True_Label"] == 1).sum()),
                "Recoverable_Negative_N": int((group["True_Label"] == 0).sum()),
                "API_Failed_N": int((group["Status"] == "api_failed").sum()),
                "Parse_Failed_N": int((group["Status"] == "parse_failed").sum()),
                "Invalid_Pred_Prob_N": int((group["Status"] == "invalid_pred_prob").sum()),
                "Missing_Record_N": int((group["Status"] == "missing_record").sum()),
                "Missing_Pred_Prob_N": int((group["Status"] == "missing_pred_prob").sum()),
            }
        )

    stage_rows = []
    for (model, target, stage), group in recoverable.groupby(["Model", "Target", "Stage"], sort=False):
        stage_rows.append(
            {
                "Model": model,
                "Target": target,
                "Stage": stage,
                "Recoverable_N": len(group),
                "Recoverable_Positive_N": int((group["True_Label"] == 1).sum()),
                "Recoverable_Negative_N": int((group["True_Label"] == 0).sum()),
                "API_Failed_N": int((group["Status"] == "api_failed").sum()),
                "Parse_Failed_N": int((group["Status"] == "parse_failed").sum()),
                "Invalid_Pred_Prob_N": int((group["Status"] == "invalid_pred_prob").sum()),
            }
        )

    pd.DataFrame(target_rows).to_csv(AUDIT_DIR / "llm_retry_gap_by_model_target.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(AUDIT_DIR / "llm_retry_gap_by_model_target_stage.csv", index=False)


if __name__ == "__main__":
    main()
