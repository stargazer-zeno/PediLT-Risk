from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from significance_common import (
    RESULTS_DIR,
    TARGETS,
    bh_fdr,
    ensure_output_dirs,
    get_model_pairs,
    get_paired_predictions,
    load_predictions,
)


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    if label_1_count == 0 or label_1_count == len(y_true):
        return {"auc_a": np.nan, "auc_b": np.nan, "delta": np.nan, "z": np.nan, "p_value": np.nan, "variance": np.nan}
    preds = np.vstack([pred_a, pred_b])[:, order]
    aucs, cov = fast_delong(preds, label_1_count)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ cov @ contrast.T).item())
    delta = float(aucs[0] - aucs[1])
    if variance <= 0:
        z = np.nan
        p_value = np.nan
    else:
        z = delta / np.sqrt(variance)
        p_value = float(2.0 * stats.norm.sf(abs(z)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta": delta,
        "z": float(z),
        "p_value": p_value,
        "variance": variance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired DeLong AUROC sensitivity tests.")
    parser.add_argument("--pairs", choices=["primary", "all"], default="all")
    args = parser.parse_args()

    ensure_output_dirs()
    predictions = load_predictions()
    rows = []
    for model_a, model_b in get_model_pairs(args.pairs):
        for target in TARGETS:
            paired = get_paired_predictions(predictions, model_a, model_b, target)
            if paired.empty:
                continue
            result = delong_test(
                paired["True_Label"].to_numpy(dtype=int),
                paired["Pred_Prob_A"].to_numpy(dtype=float),
                paired["Pred_Prob_B"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "Target": target,
                    "N_Nodes": int(len(paired)),
                    "N_Patients": int(paired["Patient_ID"].nunique()),
                    "Positive_N": int(paired["True_Label"].sum()),
                    "Negative_N": int(len(paired) - paired["True_Label"].sum()),
                    "Model_A_AUROC": result["auc_a"],
                    "Model_B_AUROC": result["auc_b"],
                    "Delta_A_minus_B": result["delta"],
                    "Z": result["z"],
                    "P_Value": result["p_value"],
                    "Variance": result["variance"],
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Q_Value_BH"] = bh_fdr(df["P_Value"])
    out_path = RESULTS_DIR / "delong_auc_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
