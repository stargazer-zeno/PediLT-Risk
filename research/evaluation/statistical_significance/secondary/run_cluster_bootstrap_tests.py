from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from significance_common import (
    METRICS,
    RESULTS_DIR,
    TARGETS,
    WeightedMetricComputer,
    bh_fdr,
    ensure_output_dirs,
    get_model_pairs,
    get_paired_predictions,
    load_predictions,
    patient_codes,
)


def bootstrap_pair(
    paired: pd.DataFrame,
    model_a: str,
    model_b: str,
    target: str,
    metrics: tuple[str, ...],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    y = paired["True_Label"].to_numpy(dtype=float)
    score_a = paired["Pred_Prob_A"].to_numpy(dtype=float)
    score_b = paired["Pred_Prob_B"].to_numpy(dtype=float)
    codes, unique_patients = patient_codes(paired["Patient_ID"])
    n_patients = len(unique_patients)
    base_weights = np.ones(len(paired), dtype=float)
    comp_a = WeightedMetricComputer(y, score_a)
    comp_b = WeightedMetricComputer(y, score_b)
    rng = np.random.default_rng(seed)

    point_estimates = {}
    for metric in metrics:
        value_a = comp_a.metric(metric, base_weights)
        value_b = comp_b.metric(metric, base_weights)
        point_estimates[metric] = (value_a, value_b, value_a - value_b)

    deltas = {metric: [] for metric in metrics}
    valid_counts = {metric: 0 for metric in metrics}
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n_patients, size=n_patients)
        patient_weight = np.bincount(sampled, minlength=n_patients).astype(float)
        weights = patient_weight[codes]
        if weights.sum() <= 0:
            continue
        for metric in metrics:
            value_a = comp_a.metric(metric, weights)
            value_b = comp_b.metric(metric, weights)
            if np.isfinite(value_a) and np.isfinite(value_b):
                deltas[metric].append(value_a - value_b)
                valid_counts[metric] += 1

    rows = []
    positives = int(y.sum())
    for metric in metrics:
        dist = np.asarray(deltas[metric], dtype=float)
        value_a, value_b, delta = point_estimates[metric]
        if len(dist):
            ci_low, ci_high = np.percentile(dist, [2.5, 97.5])
            p_lower = np.mean(dist <= 0)
            p_upper = np.mean(dist >= 0)
            p_value = min(1.0, 2.0 * min(p_lower, p_upper))
            bootstrap_mean = float(np.mean(dist))
            bootstrap_sd = float(np.std(dist, ddof=1)) if len(dist) > 1 else np.nan
        else:
            ci_low = ci_high = p_value = bootstrap_mean = bootstrap_sd = np.nan
        rows.append(
            {
                "Model_A": model_a,
                "Model_B": model_b,
                "Target": target,
                "Metric": metric,
                "N_Nodes": int(len(paired)),
                "N_Patients": int(n_patients),
                "Positive_N": positives,
                "Negative_N": int(len(paired) - positives),
                "Model_A_Value": value_a,
                "Model_B_Value": value_b,
                "Delta_A_minus_B": delta,
                "Bootstrap_Mean_Delta": bootstrap_mean,
                "Bootstrap_SD_Delta": bootstrap_sd,
                "CI_95_Low": float(ci_low),
                "CI_95_High": float(ci_high),
                "P_Value": float(p_value),
                "N_Bootstrap_Requested": int(n_bootstrap),
                "N_Bootstrap_Valid": int(valid_counts[metric]),
                "Seed": int(seed),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired patient-cluster bootstrap tests.")
    parser.add_argument("--pairs", choices=["primary", "all"], default="primary")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    ensure_output_dirs()
    predictions = load_predictions()
    pairs = get_model_pairs(args.pairs)
    all_rows: list[dict[str, object]] = []
    started = time.time()
    job_index = 0
    for model_a, model_b in pairs:
        for target in TARGETS:
            paired = get_paired_predictions(predictions, model_a, model_b, target)
            if paired.empty or paired["True_Label"].nunique() < 2:
                continue
            job_index += 1
            seed = args.seed + job_index * 1009
            print(
                f"[{job_index}] {model_a} vs {model_b}, {target}: "
                f"n={len(paired)}, patients={paired['Patient_ID'].nunique()}, seed={seed}",
                flush=True,
            )
            rows = bootstrap_pair(
                paired=paired,
                model_a=model_a,
                model_b=model_b,
                target=target,
                metrics=tuple(args.metrics),
                n_bootstrap=args.n_bootstrap,
                seed=seed,
            )
            all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    if not result.empty:
        auroc_mask = result["Metric"] == "auroc"
        result.loc[auroc_mask, "Q_Value_BH_All_AUROC"] = bh_fdr(result.loc[auroc_mask, "P_Value"])

    prefix = args.output_prefix or f"{args.pairs}_cluster_bootstrap"
    out_path = RESULTS_DIR / f"{prefix}.csv"
    result.to_csv(out_path, index=False)

    if args.pairs == "primary":
        primary_auc = result[result["Metric"] == "auroc"].copy()
        secondary = result[result["Metric"] != "auroc"].copy()
        primary_auc.to_csv(RESULTS_DIR / "primary_cluster_bootstrap_auc.csv", index=False)
        secondary.to_csv(RESULTS_DIR / "secondary_cluster_bootstrap_metrics.csv", index=False)
    elif args.pairs == "all":
        all_auc = result[result["Metric"] == "auroc"].copy()
        all_auc.to_csv(RESULTS_DIR / "all_pairwise_auc_fdr.csv", index=False)

    elapsed = time.time() - started
    print(f"Wrote {out_path}")
    print(f"Elapsed seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
