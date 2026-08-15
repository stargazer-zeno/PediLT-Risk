import json
from pathlib import Path
import sys

import pandas as pd

PREPROCESS_DIR = Path(__file__).resolve().parents[1] / "preprocessing"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from sequence_common import TARGETS, TRAIN_ROOT, format_metric


SUMMARY_DIR = TRAIN_ROOT / "summary"

METRIC_FILES = {
    "XGBoost_Sequence": TRAIN_ROOT / "xgboost" / "xgb_sequence_metrics.json",
    "LSTM_Sequence": TRAIN_ROOT / "lstm" / "lstm_sequence_metrics.json",
    "RSF_Sequence": TRAIN_ROOT / "rsf" / "rsf_sequence_metrics.json",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_markdown(df):
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                value = ""
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, path in METRIC_FILES.items():
        if not path.exists():
            rows.append({"Model": model_name, "Target": "MISSING", "Status": f"missing {path}"})
            continue
        metrics = load_json(path)
        for target in TARGETS:
            m = metrics.get("test", {}).get(target, {})
            rows.append(
                {
                    "Model": model_name,
                    "Target": target,
                    "AUROC": m.get("auroc"),
                    "N": m.get("n"),
                    "Positive_N": m.get("positive_n"),
                    "Status": "ok",
                }
            )
        if "group_oof" in metrics:
            for target in TARGETS:
                m = metrics.get("group_oof", {}).get(target, {})
                rows.append(
                    {
                        "Model": f"{model_name}_GroupOOF",
                        "Target": target,
                        "AUROC": m.get("auroc"),
                        "N": m.get("n"),
                        "Positive_N": m.get("positive_n"),
                        "Status": "ok",
                    }
                )

    df = pd.DataFrame(rows)

    csv_path = SUMMARY_DIR / "metrics_summary.csv"
    md_path = SUMMARY_DIR / "metrics_summary.md"
    df.to_csv(csv_path, index=False)

    md_df = df.copy()
    for col in ["AUROC"]:
        if col in md_df.columns:
            md_df[col] = md_df[col].map(format_metric)
    md_path.write_text(to_markdown(md_df), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
