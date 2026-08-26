#!/usr/bin/env python3
"""Run patient-disjoint Platt calibration on authorized prediction files.

This public script never reads a bundled clinical dataset.  Supply authorized
paths in an input manifest and write all generated, patient-level outputs to a
directory outside this repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from calibration_common import (
    EPS,
    TARGETS,
    VARIANTS,
    calibration_metrics,
    decision_curve,
    ensure_external_output,
    fit_platt,
    load_manifest,
    manifest_path,
    model_manifest,
    normalize_predictions,
    patient_cluster_bootstrap,
    patient_summary,
    recall80_threshold,
    select_patient_split,
    static_feature_frame,
    write_json,
)


MODULE_ROOT = Path(__file__).resolve().parent
COLORS = {
    "Original_uncalibrated": "#6b7280",
    "Platt_calibrated": "#2563eb",
    "Isotonic_sensitivity": "#16a34a",
}
LABELS = {
    "Original_uncalibrated": "Original",
    "Platt_calibrated": "Platt",
    "Isotonic_sensitivity": "Isotonic (sensitivity)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient-disjoint probability calibration for authorized data.")
    parser.add_argument("--input-manifest", required=True, type=Path, help="Private JSON manifest; never commit it.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Private directory outside this repository.")
    parser.add_argument("--calibration-patients", type=int, default=392)
    parser.add_argument("--split-candidates", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=20260824)
    parser.add_argument("--bootstrap-n", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--min-positive-patients", type=int, default=20)
    parser.add_argument("--skip-model-verification", action="store_true")
    return parser.parse_args()


def _resolve_model_paths(manifest: dict) -> dict[str, str] | None:
    raw_paths = manifest.get("model_paths")
    if raw_paths is None:
        return None
    if not isinstance(raw_paths, dict):
        raise ValueError("model_paths must be an object mapping target labels to paths")
    resolved: dict[str, str] = {}
    root = Path(manifest["_manifest_path"]).parent
    for target, value in raw_paths.items():
        path = Path(str(value))
        resolved[str(target)] = str((root / path).resolve() if not path.is_absolute() else path.resolve())
    return resolved


def fit_calibrators(source: pd.DataFrame, calibration_patients: set[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    calibrators: dict = {}
    threshold_rows: list[dict] = []
    output_frames: list[pd.DataFrame] = []
    for target in TARGETS:
        group = source.loc[
            (source["Target"] == target) & source["Patient_ID"].isin(calibration_patients)
        ].copy()
        if group.empty or group["True_Label"].nunique() < 2:
            raise ValueError(f"Calibration source has insufficient outcome variation for {target}")
        raw = np.clip(group["Pred_Prob"].to_numpy(dtype=float), EPS, 1 - EPS)
        parameters, platt = fit_platt(group["True_Label"].to_numpy(dtype=int), raw)
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, group["True_Label"])
        isotonic_probability = isotonic.predict(raw)
        parameters.update(
            {
                "target": target,
                "calibration_nodes": int(len(group)),
                "calibration_patients": int(group["Patient_ID"].nunique()),
                "calibration_positive_nodes": int(group["True_Label"].sum()),
                "calibration_positive_patients": int(
                    group.loc[group["True_Label"] == 1, "Patient_ID"].nunique()
                ),
            }
        )
        calibrators[target] = {"parameters": parameters, "isotonic": isotonic}
        for variant, probability in (
            ("Original_uncalibrated", raw),
            ("Platt_calibrated", platt),
            ("Isotonic_sensitivity", isotonic_probability),
        ):
            threshold_rows.append(
                {
                    "Dataset": "calibration",
                    "Variant": variant,
                    "Target": target,
                    "N": int(len(group)),
                    "Patients": int(group["Patient_ID"].nunique()),
                    "Recall80_Threshold": recall80_threshold(group["True_Label"].to_numpy(dtype=int), probability),
                }
            )
        output = group[["Target", "Sample_ID", "Patient_ID", "Visit_Count", "Stage", "True_Label"]].copy()
        output["Raw_Prob"] = raw
        output["Margin"] = logit(raw)
        output["Platt_Prob"] = platt
        output["Isotonic_Prob"] = isotonic_probability
        output_frames.append(output)
    return calibrators, pd.DataFrame(threshold_rows), pd.concat(output_frames, ignore_index=True)


def apply_calibrators(retained: pd.DataFrame, retained_patients: set[str], calibrators: dict) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for target in TARGETS:
        group = retained.loc[
            (retained["Target"] == target) & retained["Patient_ID"].isin(retained_patients)
        ].copy()
        if group.empty or group["True_Label"].nunique() < 2:
            raise ValueError(f"Retained evaluation data has insufficient outcome variation for {target}")
        raw = np.clip(group["Pred_Prob"].to_numpy(dtype=float), EPS, 1 - EPS)
        parameters = calibrators[target]["parameters"]
        group["Raw_Prob"] = raw
        group["Margin"] = logit(raw)
        group["Platt_Prob"] = expit(parameters["intercept"] + parameters["slope"] * group["Margin"])
        group["Isotonic_Prob"] = calibrators[target]["isotonic"].predict(raw)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def evaluate_retained(retained: pd.DataFrame, bootstrap_n: int, bootstrap_seed: int) -> dict[str, pd.DataFrame]:
    overall_rows: list[dict] = []
    calibration_rows: list[dict] = []
    bins: list[pd.DataFrame] = []
    dca_tables: list[pd.DataFrame] = []
    dca_summary_rows: list[dict] = []
    stage_rows: list[dict] = []
    for target in TARGETS:
        target_frame = retained.loc[retained["Target"] == target]
        for variant, column in (
            ("Original_uncalibrated", "Raw_Prob"),
            ("Platt_calibrated", "Platt_Prob"),
            ("Isotonic_sensitivity", "Isotonic_Prob"),
        ):
            y = target_frame["True_Label"].to_numpy(dtype=int)
            probability = target_frame[column].to_numpy(dtype=float)
            discrimination, calibration = _metrics(y, probability)
            overall_rows.append(
                {
                    "Variant": variant,
                    "Target": target,
                    "N_Nodes": int(len(target_frame)),
                    "N_Patients": int(target_frame["Patient_ID"].nunique()),
                    "Positive_Nodes": int(y.sum()),
                    **discrimination,
                }
            )
            calibration_rows.append({"Variant": variant, "Target": target, **calibration})
            _, bin_table = calibration_metrics(y, probability)
            bin_table.insert(0, "Target", target)
            bin_table.insert(0, "Variant", variant)
            bins.append(bin_table)
            dca_table, dca_summary = decision_curve(y, probability, target, variant)
            dca_tables.append(dca_table)
            dca_summary_rows.append(dca_summary)
            for stage, stage_frame in target_frame.groupby("Stage", dropna=False):
                stage_y = stage_frame["True_Label"].to_numpy(dtype=int)
                if len(np.unique(stage_y)) < 2:
                    continue
                stage_metrics, _ = _metrics(stage_y, stage_frame[column].to_numpy(dtype=float))
                stage_rows.append(
                    {
                        "Variant": variant,
                        "Target": target,
                        "Stage": str(stage),
                        "N_Nodes": int(len(stage_frame)),
                        "N_Patients": int(stage_frame["Patient_ID"].nunique()),
                        "Positive_Nodes": int(stage_y.sum()),
                        **stage_metrics,
                    }
                )
    overall = pd.DataFrame(overall_rows)
    calibration = pd.DataFrame(calibration_rows)
    return {
        "overall": overall,
        "calibration": calibration,
        "bins": pd.concat(bins, ignore_index=True),
        "dca": pd.concat(dca_tables, ignore_index=True),
        "dca_summary": pd.DataFrame(dca_summary_rows),
        "stage": pd.DataFrame(stage_rows),
        "paired": patient_cluster_bootstrap(retained, bootstrap_n, bootstrap_seed),
    }


def _metrics(y: np.ndarray, probability: np.ndarray) -> tuple[dict, dict]:
    from calibration_common import discrimination_metrics

    discrimination = discrimination_metrics(y, probability)
    calibration, _ = calibration_metrics(y, probability)
    return discrimination, calibration


def comparison_tables(results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    overall_rows: list[dict] = []
    calibration_rows: list[dict] = []
    dca_rows: list[dict] = []
    for target in TARGETS:
        original = results["overall"].loc[
            (results["overall"]["Target"] == target) & (results["overall"]["Variant"] == "Original_uncalibrated")
        ].iloc[0]
        platt = results["overall"].loc[
            (results["overall"]["Target"] == target) & (results["overall"]["Variant"] == "Platt_calibrated")
        ].iloc[0]
        overall_rows.append(
            {
                "Target": target,
                "Original_AUROC": original["AUROC"],
                "Platt_AUROC": platt["AUROC"],
                "Original_AUPRC": original["AUPRC"],
                "Platt_AUPRC": platt["AUPRC"],
                "Original_Brier": original["Brier_Score"],
                "Platt_Brier": platt["Brier_Score"],
                "Delta_Platt_Minus_Original_Brier": platt["Brier_Score"] - original["Brier_Score"],
            }
        )
        original_cal = results["calibration"].loc[
            (results["calibration"]["Target"] == target) & (results["calibration"]["Variant"] == "Original_uncalibrated")
        ].iloc[0]
        platt_cal = results["calibration"].loc[
            (results["calibration"]["Target"] == target) & (results["calibration"]["Variant"] == "Platt_calibrated")
        ].iloc[0]
        calibration_rows.append(
            {
                "Target": target,
                "Original_ECE": original_cal["ECE"],
                "Platt_ECE": platt_cal["ECE"],
                "Original_Intercept": original_cal["Calibration_Intercept"],
                "Platt_Intercept": platt_cal["Calibration_Intercept"],
                "Original_Slope": original_cal["Calibration_Slope"],
                "Platt_Slope": platt_cal["Calibration_Slope"],
            }
        )
        for variant in VARIANTS:
            row = results["dca_summary"].loc[
                (results["dca_summary"]["Target"] == target) & (results["dca_summary"]["Variant"] == variant)
            ].iloc[0]
            dca_rows.append(dict(row))
    return {
        "overall_comparison": pd.DataFrame(overall_rows),
        "calibration_comparison": pd.DataFrame(calibration_rows),
        "dca_comparison": pd.DataFrame(dca_rows),
    }


def plot_results(results: dict[str, pd.DataFrame], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, target in zip(axes, TARGETS):
        axis.plot([0, 1], [0, 1], "--", color="black", lw=1, label="Ideal")
        for variant in VARIANTS:
            group = results["bins"].loc[
                (results["bins"]["Target"] == target) & (results["bins"]["Variant"] == variant)
            ]
            axis.plot(group["Mean_Predicted_Probability"], group["Observed_Event_Rate"], "o-", ms=4,
                      color=COLORS[variant], label=LABELS[variant])
        axis.set(title=f"{target} calibration", xlabel="Mean predicted probability", ylabel="Observed event rate")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(figure_dir / "calibration_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, target in zip(axes, TARGETS):
        baseline = results["dca"].loc[
            (results["dca"]["Target"] == target) & (results["dca"]["Variant"] == "Original_uncalibrated")
        ]
        axis.plot(baseline["Threshold_Probability"] * 100, baseline["Treat_All_Net_Benefit"], "--", color="black", label="Treat all")
        axis.axhline(0, color="black", ls=":", lw=1, label="Treat none")
        for variant in VARIANTS:
            group = results["dca"].loc[
                (results["dca"]["Target"] == target) & (results["dca"]["Variant"] == variant)
            ]
            axis.plot(group["Threshold_Probability"] * 100, group["Model_Net_Benefit"], color=COLORS[variant], label=LABELS[variant])
        axis.set(title=f"{target} decision curve", xlabel="Threshold probability (%)", ylabel="Net benefit")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(figure_dir / "decision_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(TARGETS))
    width = 0.25
    for index, variant in enumerate(VARIANTS):
        group = results["calibration"].loc[results["calibration"]["Variant"] == variant].set_index("Target").loc[list(TARGETS)]
        axes[0].bar(x + (index - 1) * width, group["Brier_Score"], width, label=LABELS[variant], color=COLORS[variant])
        axes[1].bar(x + (index - 1) * width, group["ECE"], width, label=LABELS[variant], color=COLORS[variant])
    for axis, title in zip(axes, ("Brier score", "Expected calibration error")):
        axis.set(title=title, xticks=x, xticklabels=TARGETS)
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(figure_dir / "calibration_metric_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.input_manifest.resolve())
    output_dir = ensure_external_output(args.output_dir, MODULE_ROOT)
    source = normalize_predictions(manifest_path(manifest, "calibration_source_predictions", required=True), manifest.get("model_name"))
    retained_source = normalize_predictions(manifest_path(manifest, "retained_evaluation_predictions", required=True), manifest.get("model_name"))
    summary = patient_summary(source)
    static, names = static_feature_frame(manifest, summary["Patient_ID"])
    split, balance, event_balance, split_summary = select_patient_split(
        summary, static, names, args.calibration_patients, args.split_candidates, args.split_seed, args.min_positive_patients
    )
    calibration_patients = set(split.loc[split["Assignment"] == "calibration", "Patient_ID"])
    retained_patients = set(split.loc[split["Assignment"] == "retained_test", "Patient_ID"])
    calibrators, thresholds, calibration_predictions = fit_calibrators(source, calibration_patients)
    retained_predictions = apply_calibrators(retained_source, retained_patients, calibrators)
    results = evaluate_retained(retained_predictions, args.bootstrap_n, args.bootstrap_seed)
    comparisons = comparison_tables(results)

    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    parameter_dir = output_dir / "parameters"
    table_dir.mkdir(parents=True, exist_ok=True)
    parameter_dir.mkdir(parents=True, exist_ok=True)
    split.to_csv(output_dir / "split_assignments.csv", index=False)
    calibration_predictions.to_csv(output_dir / "calibration_predictions.csv", index=False)
    retained_predictions.to_csv(output_dir / "retained_predictions.csv", index=False)
    thresholds.to_csv(table_dir / "calibration_thresholds.csv", index=False)
    balance.to_csv(table_dir / "static_balance.csv", index=False)
    event_balance.to_csv(table_dir / "event_followup_balance.csv", index=False)
    for name, table in {**results, **comparisons}.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)
    plot_results(results, figure_dir)

    parameters = {
        "study_role": "research reproduction only; not a deployment configuration",
        "targets": {target: calibrators[target]["parameters"] for target in TARGETS},
    }
    write_json(parameter_dir / "platt_parameters.json", parameters)
    fingerprints = {} if args.skip_model_verification else model_manifest(_resolve_model_paths(manifest))
    verification = {
        "patient_overlap_zero": not bool(calibration_patients & retained_patients),
        "probabilities_finite": bool(np.isfinite(retained_predictions[["Raw_Prob", "Platt_Prob", "Isotonic_Prob"]].to_numpy()).all()),
        "probabilities_in_unit_interval": bool(((retained_predictions[["Raw_Prob", "Platt_Prob", "Isotonic_Prob"]] >= 0) & (retained_predictions[["Raw_Prob", "Platt_Prob", "Isotonic_Prob"]] <= 1)).all().all()),
        "model_manifest": fingerprints,
    }
    write_json(output_dir / "split_summary.json", split_summary)
    write_json(output_dir / "verification_results.json", verification)
    write_json(output_dir / "model_manifest.json", fingerprints)
    print(json.dumps({"output_dir": str(output_dir), "split_summary": split_summary, "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
