#!/usr/bin/env python3
"""Independently check private outputs produced by run_calibration_analysis.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from calibration_common import (
    EPS,
    TARGETS,
    calibration_metrics,
    decision_curve,
    discrimination_metrics,
    json_ready,
    load_manifest,
    manifest_path,
    model_manifest,
    normalize_predictions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify private calibration-analysis outputs.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-manifest", type=Path, help="Optional private manifest for source and model checks.")
    parser.add_argument("--skip-model-verification", action="store_true")
    return parser.parse_args()


def _resolve_model_paths(manifest: dict) -> dict[str, str] | None:
    raw_paths = manifest.get("model_paths")
    if raw_paths is None:
        return None
    root = Path(manifest["_manifest_path"]).parent
    return {
        str(target): str(((root / Path(str(value))).resolve() if not Path(str(value)).is_absolute() else Path(str(value)).resolve()))
        for target, value in raw_paths.items()
    }


def _max_error(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    values = np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
    return float(np.max(values)) if len(values) else math.inf


def _source_match(generated: pd.DataFrame, source: pd.DataFrame) -> tuple[bool, float]:
    keys = ["Target", "Sample_ID", "Patient_ID", "True_Label"]
    merged = generated.drop(columns=["Pred_Prob"], errors="ignore").merge(
        source[keys + ["Pred_Prob"]], on=keys, how="inner", validate="one_to_one"
    )
    error = _max_error(merged["Raw_Prob"], np.clip(merged["Pred_Prob"], EPS, 1 - EPS))
    return len(merged) == len(generated) and error <= 1e-12, error


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    split = pd.read_csv(output_dir / "split_assignments.csv", dtype={"Patient_ID": str})
    calibration = pd.read_csv(output_dir / "calibration_predictions.csv", dtype={"Patient_ID": str, "Sample_ID": str})
    retained = pd.read_csv(output_dir / "retained_predictions.csv", dtype={"Patient_ID": str, "Sample_ID": str})
    parameters = json.loads((output_dir / "parameters" / "platt_parameters.json").read_text(encoding="utf-8"))
    overall = pd.read_csv(output_dir / "tables" / "overall.csv")
    calibration_table = pd.read_csv(output_dir / "tables" / "calibration.csv")
    dca_table = pd.read_csv(output_dir / "tables" / "dca_summary.csv")
    checks: list[dict] = []

    calibration_patients = set(split.loc[split["Assignment"] == "calibration", "Patient_ID"])
    retained_patients = set(split.loc[split["Assignment"] == "retained_test", "Patient_ID"])
    checks.append({"check": "patient_overlap_zero", "pass": not bool(calibration_patients & retained_patients)})
    checks.append(
        {
            "check": "output_assignments_match_prediction_files",
            "pass": set(calibration["Patient_ID"]) <= calibration_patients and set(retained["Patient_ID"]) <= retained_patients,
        }
    )
    probability_columns = ["Raw_Prob", "Platt_Prob", "Isotonic_Prob"]
    values = retained[probability_columns].to_numpy(dtype=float)
    checks.append({"check": "probabilities_finite", "pass": bool(np.isfinite(values).all())})
    checks.append({"check": "probabilities_in_unit_interval", "pass": bool(((values >= 0) & (values <= 1)).all())})

    max_platt_error = 0.0
    max_metric_error = 0.0
    max_dca_error = 0.0
    discrimination_unchanged = True
    for target in TARGETS:
        record = parameters["targets"][target]
        for frame in (calibration, retained):
            group = frame.loc[frame["Target"] == target]
            reproduced = expit(record["intercept"] + record["slope"] * group["Margin"].to_numpy(dtype=float))
            max_platt_error = max(max_platt_error, _max_error(reproduced, group["Platt_Prob"]))
        group = retained.loc[retained["Target"] == target]
        y = group["True_Label"].to_numpy(dtype=int)
        for variant, column in (
            ("Original_uncalibrated", "Raw_Prob"),
            ("Platt_calibrated", "Platt_Prob"),
            ("Isotonic_sensitivity", "Isotonic_Prob"),
        ):
            discrimination = discrimination_metrics(y, group[column].to_numpy(dtype=float))
            calibration_result, _ = calibration_metrics(y, group[column].to_numpy(dtype=float))
            saved_overall = overall.loc[(overall["Target"] == target) & (overall["Variant"] == variant)].iloc[0]
            saved_calibration = calibration_table.loc[
                (calibration_table["Target"] == target) & (calibration_table["Variant"] == variant)
            ].iloc[0]
            max_metric_error = max(
                max_metric_error,
                abs(discrimination["AUROC"] - saved_overall["AUROC"]),
                abs(discrimination["AUPRC"] - saved_overall["AUPRC"]),
                abs(discrimination["Brier_Score"] - saved_overall["Brier_Score"]),
                abs(calibration_result["ECE"] - saved_calibration["ECE"]),
            )
            _, dca = decision_curve(y, group[column].to_numpy(dtype=float), target, variant)
            saved_dca = dca_table.loc[(dca_table["Target"] == target) & (dca_table["Variant"] == variant)].iloc[0]
            max_dca_error = max(
                max_dca_error,
                abs(float(dca["Clinically_Useful_Threshold_N"]) - float(saved_dca["Clinically_Useful_Threshold_N"])),
                abs(float(dca["Max_Model_Net_Benefit"]) - float(saved_dca["Max_Model_Net_Benefit"])),
            )
        raw = overall.loc[(overall["Target"] == target) & (overall["Variant"] == "Original_uncalibrated")].iloc[0]
        platt = overall.loc[(overall["Target"] == target) & (overall["Variant"] == "Platt_calibrated")].iloc[0]
        discrimination_unchanged &= abs(raw["AUROC"] - platt["AUROC"]) <= 1e-12 and abs(raw["AUPRC"] - platt["AUPRC"]) <= 1e-12
    checks.extend(
        [
            {"check": "platt_parameters_reproduce_probabilities", "pass": max_platt_error <= 1e-12, "max_error": max_platt_error},
            {"check": "aggregate_metrics_reproducible", "pass": max_metric_error <= 1e-12, "max_error": max_metric_error},
            {"check": "decision_curve_summary_reproducible", "pass": max_dca_error <= 1e-12, "max_error": max_dca_error},
            {"check": "platt_discrimination_unchanged", "pass": bool(discrimination_unchanged)},
            {"check": "no_isotonic_serialization", "pass": not any(output_dir.rglob("*.joblib"))},
        ]
    )

    if args.input_manifest:
        manifest = load_manifest(args.input_manifest.resolve())
        source = normalize_predictions(manifest_path(manifest, "calibration_source_predictions", required=True), manifest.get("model_name"))
        retained_source = normalize_predictions(manifest_path(manifest, "retained_evaluation_predictions", required=True), manifest.get("model_name"))
        source_ok, source_error = _source_match(calibration, source)
        retained_ok, retained_error = _source_match(retained, retained_source)
        checks.extend(
            [
                {"check": "calibration_predictions_match_source", "pass": source_ok, "max_error": source_error},
                {"check": "retained_predictions_match_source", "pass": retained_ok, "max_error": retained_error},
            ]
        )
        train_ids_path = manifest_path(manifest, "training_patient_ids_csv")
        if train_ids_path:
            train_ids = pd.read_csv(train_ids_path, dtype={"Patient_ID": str})
            if "Patient_ID" not in train_ids.columns:
                raise ValueError("training_patient_ids_csv must contain Patient_ID")
            training_patients = set(train_ids["Patient_ID"].astype(str))
            checks.append(
                {
                    "check": "training_patient_overlap_zero",
                    "pass": not bool(training_patients & calibration_patients or training_patients & retained_patients),
                }
            )
        if not args.skip_model_verification and manifest.get("model_paths"):
            observed = model_manifest(_resolve_model_paths(manifest))
            saved = json.loads((output_dir / "model_manifest.json").read_text(encoding="utf-8"))
            checks.append({"check": "model_hashes_unchanged", "pass": observed == saved})

    result = {"status": "PASS" if all(item["pass"] for item in checks) else "FAIL", "checks": checks}
    write_json(output_dir / "independent_verification_results.json", result)
    print(json.dumps(json_ready(result), indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
