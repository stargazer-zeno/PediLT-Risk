from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from significance_common import (
    RESULTS_DIR,
    TARGETS,
    auc_or_nan,
    bh_fdr,
    bootstrap_auc_deltas_fast,
    ensure_output_dirs,
    get_paired_predictions,
    load_predictions,
    make_model_inventory,
    model_pairs_all,
    patient_codes,
)


def bootstrap_auc_pair(
    paired: pd.DataFrame,
    model_a: str,
    model_b: str,
    target: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    y = paired["True_Label"].to_numpy(dtype=float)
    score_a = paired["Pred_Prob_A"].to_numpy(dtype=float)
    score_b = paired["Pred_Prob_B"].to_numpy(dtype=float)
    _, unique_patients = patient_codes(paired["Patient_ID"])
    weights0 = np.ones(len(paired), dtype=float)
    auc_a = auc_or_nan(y, score_a, weights0)
    auc_b = auc_or_nan(y, score_b, weights0)
    dist = bootstrap_auc_deltas_fast(y, score_a, score_b, paired["Patient_ID"], n_bootstrap, seed)
    if len(dist):
        ci_low, ci_high = np.percentile(dist, [2.5, 97.5])
        p_value = min(1.0, 2.0 * min(float(np.mean(dist <= 0)), float(np.mean(dist >= 0))))
        boot_mean = float(np.mean(dist))
        boot_sd = float(np.std(dist, ddof=1)) if len(dist) > 1 else np.nan
    else:
        ci_low = ci_high = p_value = boot_mean = boot_sd = np.nan
    positives = int(np.sum(y))
    return {
        "Scope": "Overall",
        "Model_A": model_a,
        "Model_B": model_b,
        "Target": target,
        "Metric": "auroc",
        "N_Nodes": int(len(paired)),
        "N_Patients": int(len(unique_patients)),
        "Positive_N": positives,
        "Negative_N": int(len(paired) - positives),
        "Model_A_Value": auc_a,
        "Model_B_Value": auc_b,
        "Delta_A_minus_B": auc_a - auc_b if np.isfinite(auc_a) and np.isfinite(auc_b) else np.nan,
        "Bootstrap_Mean_Delta": boot_mean,
        "Bootstrap_SD_Delta": boot_sd,
        "CI_95_Low": float(ci_low),
        "CI_95_High": float(ci_high),
        "P_Value": float(p_value),
        "N_Bootstrap_Requested": int(n_bootstrap),
        "N_Bootstrap_Valid": int(len(dist)),
        "Seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run overall all-pairwise bootstrap AUROC tests using common-cohort predictions.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--output", default="overall_pairwise_cluster_bootstrap_auc_final_common.csv")
    args = parser.parse_args()

    ensure_output_dirs()
    predictions = load_predictions()
    make_model_inventory(predictions).to_csv(RESULTS_DIR / "model_prediction_inventory_final_common.csv", index=False)
    rows = []
    started = time.time()
    job_index = 0
    for model_a, model_b in model_pairs_all():
        for target in TARGETS:
            paired = get_paired_predictions(predictions, model_a, model_b, target)
            if paired.empty or paired["True_Label"].nunique() < 2:
                continue
            job_index += 1
            seed = args.seed + job_index * 1009
            print(f"[{job_index}] {model_a} vs {model_b}, {target}: n={len(paired)}, patients={paired['Patient_ID'].nunique()}", flush=True)
            rows.append(bootstrap_auc_pair(paired, model_a, model_b, target, args.n_bootstrap, seed))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["Q_Value_BH"] = bh_fdr(result["P_Value"])
        result["Significant_FDR_0_05"] = result["Q_Value_BH"] < 0.05
    out_path = RESULTS_DIR / args.output
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Elapsed seconds: {time.time() - started:.1f}")


if __name__ == "__main__":
    main()
