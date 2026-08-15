import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold

PREPROCESS_DIR = Path(__file__).resolve().parents[1] / "preprocessing"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from sequence_common import (
    TARGETS,
    TRAIN_ROOT,
    format_metric,
    load_schema,
    load_sequence_npz,
    score_binary,
    set_seed,
    write_json,
)


OUT_DIR = TRAIN_ROOT / "xgboost"
PRED_CSV = OUT_DIR / "xgb_sequence_saved_test_predictions.csv"
OOF_CSV = OUT_DIR / "xgb_sequence_train_group_oof_predictions.csv"
METRICS_JSON = OUT_DIR / "xgb_sequence_metrics.json"
REPORT_TXT = OUT_DIR / "xgb_sequence_report.txt"


def build_flat_features(data):
    static = data["static"].astype(np.float32)
    temporal = data["temporal"].astype(np.float32)
    time_mask = data["time_mask"].astype(np.float32)
    seq_len = data["sequence_lengths"].astype(np.float32).reshape(-1, 1)
    temporal_flat = temporal.reshape(temporal.shape[0], -1)
    return np.concatenate([static, temporal_flat, time_mask, seq_len], axis=1).astype(np.float32, copy=False)


def make_model(y, args):
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    scale_pos_weight = neg / pos if pos else 1.0
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": scale_pos_weight,
        "missing": np.nan,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_bin": args.max_bin,
    }
    if args.device:
        params["device"] = args.device
    return xgb.XGBClassifier(**params)


def target_suffix(target):
    return target.split("_", 1)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=1.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-oof", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    schema = load_schema()
    train = load_sequence_npz("train")
    test = load_sequence_npz("test")

    print("Building flattened sequence features")
    x_train_all = build_flat_features(train)
    x_test_all = build_flat_features(test)
    y_train_all = train["labels"]
    y_test_all = test["labels"]
    train_sample_ids = train["sample_ids"].astype(str)
    test_sample_ids = test["sample_ids"].astype(str)
    train_patient_ids = train["patient_ids"].astype(str)
    test_patient_ids = test["patient_ids"].astype(str)
    test_visit_counts = test["visit_counts"]

    metrics = {
        "config": {
            **vars(args),
            "model": "XGBClassifier",
            "input_representation": "static features + raw timestep values flattened + timestep mask + sequence length",
            "max_seq_len": schema["max_seq_len"],
            "static_dim": len(schema["static_feature_names"]),
            "temporal_dim_per_step": len(schema["temporal_feature_names"]),
            "feature_dim": int(x_train_all.shape[1]),
            "xgboost_version": xgb.__version__,
        },
        "test": {},
    }
    if not args.skip_oof:
        metrics["group_oof"] = {}
    report_lines = [
        "XGBoost sequence-input training",
        f"feature_dim={x_train_all.shape[1]}, max_seq_len={schema['max_seq_len']}",
    ]
    test_pred_parts = []
    oof_df = pd.DataFrame({"Sample_ID": train_sample_ids, "Patient_ID": train_patient_ids}) if not args.skip_oof else None

    for target_idx, target in enumerate(TARGETS):
        print(f"Training {target}", flush=True)
        report_lines.append(f"\n===== {target} =====")
        train_mask = ~np.isnan(y_train_all[:, target_idx])
        test_mask = ~np.isnan(y_test_all[:, target_idx])
        x_train = x_train_all[train_mask]
        x_test = x_test_all[test_mask]
        y_train = y_train_all[train_mask, target_idx].astype(int)
        y_test = y_test_all[test_mask, target_idx].astype(int)
        groups = train_patient_ids[train_mask]

        report_lines.append(
            f"train n={len(y_train)}, pos={int(np.sum(y_train == 1))}; "
            f"test n={len(y_test)}, pos={int(np.sum(y_test == 1))}"
        )

        if not args.skip_oof:
            col = f"OOF_Prob_{target_suffix(target)}"
            oof_df[col] = np.nan
            oof_probs = np.full(len(y_train), np.nan, dtype=np.float32)
            splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
            fold_scores = []
            for fold, (fit_idx, valid_idx) in enumerate(splitter.split(x_train, y_train, groups), start=1):
                print(f"  {target} fold {fold}/{args.n_splits}", flush=True)
                model = make_model(y_train[fit_idx], args)
                model.fit(x_train[fit_idx], y_train[fit_idx])
                prob = model.predict_proba(x_train[valid_idx])[:, 1]
                oof_probs[valid_idx] = prob
                fold_metrics = score_binary(y_train[valid_idx], prob)
                fold_scores.append(fold_metrics["auroc"])
                line = f"fold {fold}: AUROC={format_metric(fold_metrics['auroc'])}, val_n={len(valid_idx)}"
                report_lines.append(line)
                print(f"  {line}", flush=True)

            oof_df.loc[np.where(train_mask)[0], col] = oof_probs
            metrics["group_oof"][target] = score_binary(y_train, oof_probs)
            metrics["group_oof"][target]["fold_auroc"] = fold_scores
        else:
            report_lines.append("group OOF not requested; run without --skip-oof to generate it")

        print(f"  {target} final fit", flush=True)
        final_model = make_model(y_train, args)
        final_model.fit(x_train, y_train)
        model_path = OUT_DIR / f"xgb_sequence_{target}.json"
        final_model.save_model(model_path)

        test_prob = final_model.predict_proba(x_test)[:, 1]
        metrics["test"][target] = score_binary(y_test, test_prob)
        metrics["test"][target]["model_path"] = str(model_path)
        test_line = (
            f"test AUROC={format_metric(metrics['test'][target]['auroc'])}"
        )
        report_lines.append(test_line)
        print(f"  {test_line}", flush=True)
        if not args.skip_oof:
            report_lines.append(
                f"group OOF AUROC={format_metric(metrics['group_oof'][target]['auroc'])}"
            )

        test_pred_parts.append(
            pd.DataFrame(
                {
                    "Target": target,
                    "Sample_ID": test_sample_ids[test_mask],
                    "Patient_ID": test_patient_ids[test_mask],
                    "Visit_Count": test_visit_counts[test_mask],
                    "True_Label": y_test,
                    "Pred_Prob": test_prob,
                }
            )
        )

    pd.concat(test_pred_parts, ignore_index=True).to_csv(PRED_CSV, index=False)
    if oof_df is not None:
        oof_df.to_csv(OOF_CSV, index=False)
    write_json(METRICS_JSON, metrics)
    REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
