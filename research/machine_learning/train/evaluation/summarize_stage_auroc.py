import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = REPO_ROOT / "machine_learning" / "train"
OUT_TXT = TRAIN_ROOT / "overall_and_stage_auroc_results.txt"
STAGE_IDS_JSON = Path(os.environ.get("PEDILT_STAGE_MAP", REPO_ROOT / "data" / "test_ids_by_stage.json"))
STAGE_SPLIT_CSV = Path(os.environ.get("PEDILT_STAGE_SUMMARY", REPO_ROOT / "data" / "test_ids_stage_split_summary.csv"))
OVERALL_CSV = TRAIN_ROOT / "summary" / "metrics_summary.csv"

TARGET_ORDER = ["Label_1m", "Label_1y", "Label_5y"]
MODEL_SOURCES = [
    ("XGBoost_Sequence", TRAIN_ROOT / "xgboost" / "xgb_sequence_saved_test_predictions.csv"),
    ("LSTM_Sequence", TRAIN_ROOT / "lstm" / "lstm_sequence_saved_test_predictions.csv"),
    ("RSF_Sequence", TRAIN_ROOT / "rsf" / "rsf_sequence_saved_test_predictions.csv"),
]
STAGE_META = {
    "stage1": ("Stage1", "0d-1m", "0-30d"),
    "stage2": ("Stage2", "2m-3m", "31-90d"),
    "stage3": ("Stage3", "4m-12m", "91-365d"),
    "stage4": ("Stage4", "1y-2y", "366-730d"),
    "stage5": ("Stage5", ">2y", ">730d"),
}
STAGE_ORDER = ["Stage1", "Stage2", "Stage3", "Stage4", "Stage5"]


def format_float(value):
    try:
        if np.isfinite(value):
            return f"{float(value):.4f}"
    except Exception:
        pass
    return "NA"


def format_table(df, columns):
    rows = [[str(col) for col in columns]]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row.get(col, "")
            if pd.isna(value):
                value = ""
            values.append(str(value))
        rows.append(values)
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(columns))]
    lines = []
    for idx, row in enumerate(rows):
        lines.append("  ".join(value.ljust(widths[col_idx]) for col_idx, value in enumerate(row)))
        if idx == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def load_stage_lookup():
    with STAGE_IDS_JSON.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    lookup = {}
    for key, sample_ids in raw.items():
        stage, label, day_range = STAGE_META[key]
        for sample_id in sample_ids:
            lookup[str(sample_id)] = {
                "Stage": stage,
                "Stage_Label": label,
                "Day_Range": day_range,
            }
    return lookup


def load_predictions(stage_lookup):
    frames = []
    required = ["Target", "Sample_ID", "Patient_ID", "Visit_Count", "True_Label", "Pred_Prob"]
    for model, path in MODEL_SOURCES:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, usecols=required)
        df["Model"] = model
        df["Target"] = df["Target"].astype(str)
        df["Sample_ID"] = df["Sample_ID"].astype(str)
        df["True_Label"] = pd.to_numeric(df["True_Label"], errors="coerce")
        df["Pred_Prob"] = pd.to_numeric(df["Pred_Prob"], errors="coerce")
        stage_info = df["Sample_ID"].map(stage_lookup)
        df["Stage"] = stage_info.map(lambda x: x["Stage"] if isinstance(x, dict) else pd.NA)
        df["Stage_Label"] = stage_info.map(lambda x: x["Stage_Label"] if isinstance(x, dict) else pd.NA)
        df["Day_Range"] = stage_info.map(lambda x: x["Day_Range"] if isinstance(x, dict) else pd.NA)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_stage_auroc(predictions):
    rows = []
    valid = predictions.dropna(subset=["Stage", "True_Label", "Pred_Prob"]).copy()
    valid = valid[valid["True_Label"].isin([0, 1])]
    valid = valid[valid["Target"].isin(TARGET_ORDER)]
    for model in [m for m, _ in MODEL_SOURCES]:
        for target in TARGET_ORDER:
            for stage in STAGE_ORDER:
                group = valid[
                    (valid["Model"] == model)
                    & (valid["Target"] == target)
                    & (valid["Stage"] == stage)
                ]
                positive_n = int((group["True_Label"] == 1).sum())
                negative_n = int((group["True_Label"] == 0).sum())
                if len(group) > 0 and group["True_Label"].nunique() == 2:
                    auroc = roc_auc_score(group["True_Label"], group["Pred_Prob"])
                else:
                    auroc = np.nan
                stage_label = STAGE_META[f"stage{STAGE_ORDER.index(stage) + 1}"][1]
                day_range = STAGE_META[f"stage{STAGE_ORDER.index(stage) + 1}"][2]
                rows.append(
                    {
                        "Model": model,
                        "Target": target,
                        "Stage": stage,
                        "Stage_Label": stage_label,
                        "Day_Range": day_range,
                        "AUROC": format_float(auroc),
                        "N": int(len(group)),
                        "Positive_N": positive_n,
                        "Negative_N": negative_n,
                    }
                )
    return pd.DataFrame(rows)


def load_overall_results():
    df = pd.read_csv(OVERALL_CSV)
    keep_models = [m for m, _ in MODEL_SOURCES]
    df = df[df["Model"].isin(keep_models)].copy()
    rows = []
    for model in keep_models:
        for target in TARGET_ORDER:
            row = df[(df["Model"] == model) & (df["Target"] == target)].iloc[0]
            rows.append(
                {
                    "Model": model,
                    "Target": target,
                    "Overall_AUROC": format_float(row["AUROC"]),
                    "N": int(row["N"]),
                    "Positive_N": int(row["Positive_N"]),
                }
            )
    return pd.DataFrame(rows)


def main():
    stage_lookup = load_stage_lookup()
    predictions = load_predictions(stage_lookup)
    stage_results = compute_stage_auroc(predictions)
    overall_results = load_overall_results()
    split_summary = pd.read_csv(STAGE_SPLIT_CSV)

    matched_samples = predictions.drop_duplicates(["Model", "Sample_ID"])
    coverage = matched_samples.groupby("Model")["Stage"].apply(lambda s: int(s.notna().sum())).to_dict()
    total = matched_samples.groupby("Model")["Sample_ID"].nunique().to_dict()

    lines = [
        "Overall and stage AUROC results",
        f"Output file: {OUT_TXT}",
        "",
        "Stage definition source:",
        f"- {STAGE_IDS_JSON}",
        f"- {STAGE_SPLIT_CSV}",
        "",
        "Stage split summary:",
        format_table(
            split_summary,
            ["Stage", "Stage_Label", "Day_Range", "Count", "Min_Postop_Day", "Max_Postop_Day"],
        ),
        "",
        "Stage ID coverage by model:",
    ]
    for model in [m for m, _ in MODEL_SOURCES]:
        lines.append(f"- {model}: {coverage.get(model, 0)} / {total.get(model, 0)} test nodes assigned to stage")

    lines.extend(
        [
            "",
            "Overall held-out test results:",
            format_table(overall_results, ["Model", "Target", "Overall_AUROC", "N", "Positive_N"]),
            "",
            "Stage held-out test AUROC results:",
            format_table(
                stage_results,
                ["Model", "Target", "Stage", "Stage_Label", "Day_Range", "AUROC", "N", "Positive_N", "Negative_N"],
            ),
            "",
            "Notes:",
            "- AUROC is computed within each model-target-stage group.",
            "- Rows with missing stage assignment are excluded from stage-specific AUROC.",
            "- NA means AUROC cannot be computed because the group does not contain both classes.",
        ]
    )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_TXT)


if __name__ == "__main__":
    main()
