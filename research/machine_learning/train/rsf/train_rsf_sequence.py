import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

PREPROCESS_DIR = Path(__file__).resolve().parents[1] / "preprocessing"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from sequence_common import (
    TARGETS,
    TIME_POINTS,
    TRAIN_ROOT,
    format_metric,
    load_schema,
    load_sequence_npz,
    score_binary,
    set_seed,
    write_json,
)


OUT_DIR = TRAIN_ROOT / "rsf"
PRED_CSV = OUT_DIR / "rsf_sequence_saved_test_predictions.csv"
METRICS_JSON = OUT_DIR / "rsf_sequence_metrics.json"
REPORT_TXT = OUT_DIR / "rsf_sequence_report.txt"
MODEL_PATH = OUT_DIR / "rsf_sequence_model.joblib"
IMPUTER_PATH = OUT_DIR / "rsf_sequence_imputer.joblib"


def build_flat_features(data):
    static = data["static"].astype(np.float32)
    temporal = data["temporal"].astype(np.float32)
    time_mask = data["time_mask"].astype(np.float32)
    seq_len = data["sequence_lengths"].astype(np.float32).reshape(-1, 1)
    temporal_flat = temporal.reshape(temporal.shape[0], -1)
    return np.concatenate([static, temporal_flat, time_mask, seq_len], axis=1).astype(np.float32, copy=False)


def load_survival(split):
    data = np.load(TRAIN_ROOT / "datasets" / f"survival_{split}.npz", allow_pickle=False)
    return Surv.from_arrays(event=data["event"].astype(bool), time=data["time"].astype(float))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument("--min-samples-split", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    schema = load_schema()
    train = load_sequence_npz("train")
    test = load_sequence_npz("test")

    print("Building flattened sequence features")
    x_train_raw = build_flat_features(train)
    x_test_raw = build_flat_features(test)
    print(f"feature_dim={x_train_raw.shape[1]}")

    print("Imputing missing values with train-set means")
    imputer = SimpleImputer(strategy="mean")
    x_train = imputer.fit_transform(x_train_raw).astype(np.float32)
    x_test = imputer.transform(x_test_raw).astype(np.float32)
    joblib.dump(imputer, IMPUTER_PATH)

    y_train_surv = load_survival("train")
    y_test_surv = load_survival("test")
    model = RandomSurvivalForest(
        n_estimators=args.n_estimators,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )

    print("Training RandomSurvivalForest")
    model.fit(x_train, y_train_surv)
    joblib.dump(model, MODEL_PATH)
    c_index = float(model.score(x_test, y_test_surv))
    print(f"C-index={c_index:.4f}")

    print("Predicting survival functions")
    survival_functions = model.predict_survival_function(x_test)
    y_test_all = test["labels"]
    test_sample_ids = test["sample_ids"].astype(str)
    test_patient_ids = test["patient_ids"].astype(str)
    test_visit_counts = test["visit_counts"]

    metrics = {
        "config": {
            **vars(args),
            "model": "RandomSurvivalForest",
            "input_representation": "static features + raw timestep values flattened + timestep mask + sequence length",
            "max_seq_len": schema["max_seq_len"],
            "static_dim": len(schema["static_feature_names"]),
            "temporal_dim_per_step": len(schema["temporal_feature_names"]),
            "feature_dim": int(x_train.shape[1]),
            "model_path": str(MODEL_PATH),
            "imputer_path": str(IMPUTER_PATH),
        },
        "c_index": c_index,
        "test": {},
    }
    report_lines = [
        "RSF sequence-input training",
        f"C-index={c_index:.4f}",
        f"feature_dim={x_train.shape[1]}, max_seq_len={schema['max_seq_len']}",
    ]
    pred_parts = []

    for target_idx, target in enumerate(TARGETS):
        horizon_days = TIME_POINTS[target]
        prob = np.asarray([1.0 - fn(horizon_days) for fn in survival_functions], dtype=np.float32)
        valid = ~np.isnan(y_test_all[:, target_idx])
        y_true = y_test_all[valid, target_idx].astype(int)
        y_prob = prob[valid]
        metrics["test"][target] = score_binary(y_true, y_prob)
        report_lines.append(
            f"{target}: AUROC={format_metric(metrics['test'][target]['auroc'])}, "
            f"n={metrics['test'][target]['n']}, pos={metrics['test'][target]['positive_n']}"
        )
        pred_parts.append(
            pd.DataFrame(
                {
                    "Target": target,
                    "Sample_ID": test_sample_ids[valid],
                    "Patient_ID": test_patient_ids[valid],
                    "Visit_Count": test_visit_counts[valid],
                    "True_Label": y_true,
                    "Pred_Prob": y_prob,
                }
            )
        )

    pd.concat(pred_parts, ignore_index=True).to_csv(PRED_CSV, index=False)
    write_json(METRICS_JSON, metrics)
    REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
