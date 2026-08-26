"""Shared utilities for the public probability-calibration workflow.

The functions in this module operate on authorized, patient-level files supplied
at runtime.  The repository deliberately contains no such files.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


TARGETS = ("1m", "1y", "5y")
VARIANTS = ("Original_uncalibrated", "Platt_calibrated", "Isotonic_sensitivity")
EPS = 1e-6
DCA_RANGES = {"1m": (0.001, 0.100), "1y": (0.005, 0.200), "5y": (0.010, 0.300)}
REQUIRED_PREDICTION_COLUMNS = {"Patient_ID", "Sample_ID", "Target", "True_Label", "Pred_Prob"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Input manifest must be a JSON object")
    required = {"calibration_source_predictions", "retained_evaluation_predictions"}
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"Input manifest is missing required keys: {', '.join(missing)}")
    value["_manifest_path"] = path.resolve()
    return value


def manifest_path(manifest: dict[str, Any], key: str, required: bool = False) -> Path | None:
    raw_value = manifest.get(key)
    if raw_value in (None, ""):
        if required:
            raise ValueError(f"Input manifest key '{key}' is required")
        return None
    path = Path(str(raw_value))
    if not path.is_absolute():
        path = Path(manifest["_manifest_path"]).parent / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest path for '{key}' does not exist: {path}")
    return path


def normalize_predictions(path: Path, model_name: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if model_name and "Model" in frame.columns:
        frame = frame.loc[frame["Model"].astype(str) == model_name].copy()
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["Patient_ID"] = frame["Patient_ID"].astype(str)
    frame["Sample_ID"] = frame["Sample_ID"].astype(str)
    frame["Target"] = frame["Target"].astype(str).str.replace("Label_", "", regex=False)
    unknown_targets = sorted(set(frame["Target"]) - set(TARGETS))
    if unknown_targets:
        raise ValueError(f"{path.name} has unsupported targets: {unknown_targets}")
    frame["True_Label"] = pd.to_numeric(frame["True_Label"], errors="raise").astype(int)
    if not frame["True_Label"].isin([0, 1]).all():
        raise ValueError(f"{path.name} True_Label must contain only 0 and 1")
    frame["Pred_Prob"] = pd.to_numeric(frame["Pred_Prob"], errors="raise")
    if not np.isfinite(frame["Pred_Prob"]).all():
        raise ValueError(f"{path.name} contains non-finite probabilities")
    if (frame["Pred_Prob"] < 0).any() or (frame["Pred_Prob"] > 1).any():
        raise ValueError(f"{path.name} probabilities must be in [0, 1]")
    if frame.duplicated(["Target", "Sample_ID"]).any():
        raise ValueError(f"{path.name} has duplicate Target/Sample_ID rows")
    if "Visit_Count" not in frame.columns:
        frame["Visit_Count"] = np.nan
    if "Stage" not in frame.columns:
        frame["Stage"] = "Unspecified"
    return frame


def _patient_stratum(positive: np.ndarray, eligible: np.ndarray) -> str:
    if positive.any():
        return "event_any_horizon"
    if eligible[2]:
        return "observed_5y"
    if eligible[1]:
        return "observed_1y"
    return "observed_1m"


def patient_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient_id, group in predictions.groupby("Patient_ID", sort=True):
        positive = np.asarray(
            [bool(((group["Target"] == target) & (group["True_Label"] == 1)).any()) for target in TARGETS]
        )
        eligible = np.asarray([bool((group["Target"] == target).any()) for target in TARGETS])
        row: dict[str, Any] = {
            "Patient_ID": str(patient_id),
            "Stratum": _patient_stratum(positive, eligible),
            "Node_N": int(len(group)),
        }
        for index, target in enumerate(TARGETS):
            row[f"PositivePatient_{target}"] = int(positive[index])
            row[f"EligiblePatient_{target}"] = int(eligible[index])
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No patients were available for calibration")
    return result


def static_feature_frame(manifest: dict[str, Any], patients: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Load optional patient-level static features for split-balance selection."""
    static_csv = manifest_path(manifest, "patient_static_features_csv")
    sequence_npz = manifest_path(manifest, "sequence_test_npz")
    if static_csv and sequence_npz:
        raise ValueError("Provide only one of patient_static_features_csv or sequence_test_npz")
    if static_csv:
        frame = pd.read_csv(static_csv)
        if "Patient_ID" not in frame.columns:
            raise ValueError("patient_static_features_csv must contain Patient_ID")
        frame["Patient_ID"] = frame["Patient_ID"].astype(str)
        if frame["Patient_ID"].duplicated().any():
            raise ValueError("patient_static_features_csv must contain one row per patient")
        feature_names = [
            column for column in frame.columns if column != "Patient_ID" and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not feature_names:
            raise ValueError("patient_static_features_csv has no numeric static feature columns")
        frame = frame[["Patient_ID", *feature_names]].copy()
    elif sequence_npz:
        data = np.load(sequence_npz, allow_pickle=False)
        if "patient_ids" not in data or "static" not in data:
            raise ValueError("sequence_test_npz must provide patient_ids and static arrays")
        patient_ids = data["patient_ids"].astype(str)
        static = np.asarray(data["static"], dtype=float)
        unique_ids, first_indices = np.unique(patient_ids, return_index=True)
        if static.shape[0] != len(patient_ids):
            raise ValueError("sequence_test_npz static and patient_ids lengths differ")
        feature_names = [f"Static_{index:03d}" for index in range(static.shape[1])]
        schema_path = manifest_path(manifest, "schema_json")
        if schema_path:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            names = schema.get("static_feature_names")
            if isinstance(names, list) and len(names) == static.shape[1]:
                feature_names = [str(name) for name in names]
        frame = pd.DataFrame(static[first_indices], columns=feature_names)
        frame.insert(0, "Patient_ID", unique_ids)
    else:
        return pd.DataFrame({"Patient_ID": patients.astype(str)}), []
    missing = sorted(set(patients.astype(str)) - set(frame["Patient_ID"]))
    if missing:
        raise ValueError("Static feature source is missing patients in calibration_source_predictions")
    return frame, feature_names


def standardized_differences(calibration: np.ndarray, retained: np.ndarray) -> np.ndarray:
    cal_mean = calibration.mean(axis=0)
    retained_mean = retained.mean(axis=0)
    cal_var = calibration.var(axis=0, ddof=1)
    retained_var = retained.var(axis=0, ddof=1)
    pooled = np.sqrt((cal_var + retained_var) / 2)
    return np.divide(
        np.abs(cal_mean - retained_mean),
        pooled,
        out=np.zeros_like(pooled, dtype=float),
        where=pooled > 0,
    )


def select_patient_split(
    summary: pd.DataFrame,
    static_features: pd.DataFrame,
    feature_names: list[str],
    calibration_patients: int,
    candidates: int,
    seed: int,
    min_positive_patients: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if calibration_patients <= 0 or calibration_patients >= len(summary):
        raise ValueError("calibration_patients must be between 1 and the total patient count minus one")
    strata = summary["Stratum"].to_numpy()
    if pd.Series(strata).value_counts().min() < 2:
        raise ValueError("Each patient split stratum must contain at least two patients")
    static = summary[["Patient_ID"]].merge(static_features, on="Patient_ID", how="left", validate="one_to_one")
    if feature_names:
        values = static[feature_names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Static feature values must be finite for balance selection")
        balance_names = feature_names
    else:
        balance_names = [
            *(f"PositivePatient_{target}" for target in TARGETS),
            *(f"EligiblePatient_{target}" for target in TARGETS),
        ]
        values = summary[balance_names].to_numpy(dtype=float)
    splitter = StratifiedShuffleSplit(n_splits=candidates, test_size=calibration_patients, random_state=seed)
    best: tuple[tuple[float, float, int], np.ndarray, np.ndarray, np.ndarray] | None = None
    feasible = 0
    dummy = np.zeros(len(summary))
    for candidate_index, (retained_index, calibration_index) in enumerate(splitter.split(dummy, strata), start=1):
        positive_counts = [
            int(summary.iloc[calibration_index][f"PositivePatient_{target}"].sum()) for target in TARGETS
        ]
        if min(positive_counts) < min_positive_patients:
            continue
        smd = standardized_differences(values[calibration_index], values[retained_index])
        key = (float(np.max(smd)), float(np.mean(smd)), candidate_index)
        feasible += 1
        if best is None or key < best[0]:
            best = (key, calibration_index.copy(), retained_index.copy(), smd.copy())
    if best is None:
        raise RuntimeError("No candidate split met the minimum positive-patient requirement")
    key, calibration_index, retained_index, smd = best
    assignments = np.full(len(summary), "retained_test", dtype=object)
    assignments[calibration_index] = "calibration"
    split = summary.copy()
    split["Assignment"] = assignments
    split["Candidate_Selected"] = key[2]
    split = split.sort_values(["Assignment", "Patient_ID"]).reset_index(drop=True)
    balance = pd.DataFrame(
        {
            "Variable": balance_names,
            "Calibration_Mean": values[calibration_index].mean(axis=0),
            "Retained_Test_Mean": values[retained_index].mean(axis=0),
            "Absolute_SMD": smd,
            "Pass_0.10": smd <= 0.10,
        }
    ).sort_values("Absolute_SMD", ascending=False, ignore_index=True)
    event_rows: list[dict[str, Any]] = []
    for assignment, indices in (("calibration", calibration_index), ("retained_test", retained_index)):
        subset = summary.iloc[indices]
        for target in TARGETS:
            for kind in ("PositivePatient", "EligiblePatient"):
                event_rows.append(
                    {
                        "Assignment": assignment,
                        "Measure": f"{kind}_{target}",
                        "Patient_N": int(len(subset)),
                        "Count": int(subset[f"{kind}_{target}"].sum()),
                    }
                )
    event_balance = pd.DataFrame(event_rows)
    event_balance["Fraction"] = event_balance["Count"] / event_balance["Patient_N"]
    split_summary = {
        "split_policy": "patient-disjoint stratified split selected by minimum standardized mean difference",
        "split_seed": seed,
        "candidate_count": candidates,
        "feasible_candidate_count": feasible,
        "selected_candidate": key[2],
        "patient_counts": {
            "calibration": int(len(calibration_index)),
            "retained_test": int(len(retained_index)),
            "total": int(len(summary)),
        },
        "max_absolute_smd": key[0],
        "mean_absolute_smd": key[1],
        "used_static_features": bool(feature_names),
    }
    return split, balance, event_balance, split_summary


def fit_logistic_intercept_slope(y: np.ndarray, margin: np.ndarray) -> tuple[float, float, bool, str | None]:
    y = np.asarray(y, dtype=float)
    margin = np.asarray(margin, dtype=float)
    design = np.column_stack([np.ones(len(margin)), margin])
    initial = np.asarray([logit(np.clip(y.mean(), EPS, 1 - EPS)), 1.0], dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta)), design.T @ (expit(eta) - y)

    result = minimize(
        lambda beta: objective(beta)[0],
        initial,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if len(result.x) != 2:
        return math.nan, math.nan, False, "optimizer returned an invalid parameter vector"
    return float(result.x[0]), float(result.x[1]), bool(result.success), None if result.success else str(result.message)


def fit_platt(y: np.ndarray, raw_probability: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    raw = np.clip(np.asarray(raw_probability, dtype=float), EPS, 1 - EPS)
    margin = logit(raw)
    try:
        intercept, slope, converged, reason = fit_logistic_intercept_slope(np.asarray(y, dtype=int), margin)
    except Exception as exc:  # pragma: no cover - defensive fallback
        intercept, slope, converged, reason = math.nan, math.nan, False, repr(exc)
    fallback_used = False
    if not converged or not np.isfinite(intercept) or not np.isfinite(slope) or slope <= 0:
        fallback_used = True
        if reason is None:
            reason = f"invalid Platt fit: intercept={intercept}, slope={slope}, converged={converged}"
        prevalence = float(np.asarray(y).mean())
        intercept = float(brentq(lambda value: float(np.mean(expit(value + margin)) - prevalence), -50, 50))
        slope = 1.0
    values = expit(intercept + slope * margin)
    return {
        "method": "Platt scaling on logit(clipped_raw_probability)",
        "formula": "sigmoid(intercept + slope * logit(clipped_raw_probability))",
        "probability_clip": [EPS, 1 - EPS],
        "intercept": intercept,
        "slope": slope,
        "fit_converged": bool(converged),
        "fallback_used": fallback_used,
        "fallback_reason": reason,
    }, values


def recall80_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    if np.sum(y) == 0:
        return math.nan
    order = np.argsort(-probability, kind="mergesort")
    probabilities = probability[order]
    labels = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(probabilities)) + 1]
    thresholds = probabilities[starts]
    positive_counts = np.add.reduceat(labels, starts)
    recall = np.cumsum(positive_counts) / np.sum(labels)
    eligible = np.flatnonzero(recall >= 0.80)
    return float(thresholds[eligible[0]]) if len(eligible) else math.nan


def discrimination_metrics(y: np.ndarray, probability: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(np.unique(y)) < 2:
        return {"AUROC": math.nan, "AUPRC": math.nan, "Brier_Score": math.nan}
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    return {
        "AUROC": float(roc_auc_score(y, probability, sample_weight=weights)),
        "AUPRC": float(average_precision_score(y, probability, sample_weight=weights)),
        "Brier_Score": float(np.average((probability - y) ** 2, weights=weights)),
    }


def calibration_metrics(y: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> tuple[dict[str, float], pd.DataFrame]:
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    edges = np.unique(np.quantile(probability, np.linspace(0, 1, n_bins + 1)))
    bin_codes = np.zeros(len(probability), dtype=int) if len(edges) <= 2 else np.searchsorted(edges[1:-1], probability, side="left")
    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    hl = 0.0
    for bin_index in np.unique(bin_codes):
        mask = bin_codes == bin_index
        observed = float(y[mask].mean())
        predicted = float(probability[mask].mean())
        count = int(mask.sum())
        error = abs(observed - predicted)
        ece += count / len(y) * error
        mce = max(mce, error)
        expected = count * predicted
        denominator = expected * (1 - expected / count)
        if denominator > 0:
            hl += (int(y[mask].sum()) - expected) ** 2 / denominator
        rows.append(
            {
                "Bin": int(bin_index + 1),
                "N": count,
                "Positive_N": int(y[mask].sum()),
                "Mean_Predicted_Probability": predicted,
                "Observed_Event_Rate": observed,
                "Absolute_Error": error,
            }
        )
    intercept, slope, _, _ = fit_logistic_intercept_slope(y, logit(probability))
    prevalence = float(y.mean())
    brier = float(np.mean((probability - y) ** 2))
    reference_brier = prevalence * (1 - prevalence)
    metrics = {
        "Brier_Score": brier,
        "Brier_Skill_Score": float(1 - brier / reference_brier) if reference_brier else math.nan,
        "ECE": float(ece),
        "MCE": float(mce),
        "Calibration_Intercept": intercept,
        "Calibration_Slope": slope,
        "HL_ChiSquare": float(hl),
        "HL_DF": max(len(rows) - 2, 1),
        "HL_P_Value": float(chi2.sf(hl, max(len(rows) - 2, 1))),
        "Calibration_Bin_Count": len(rows),
    }
    return metrics, pd.DataFrame(rows)


def decision_curve(y: np.ndarray, probability: np.ndarray, target: str, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    lo, hi = DCA_RANGES[target]
    thresholds = np.round(np.arange(lo, hi + 0.0005, 0.001), 6)
    prevalence = float(np.mean(y))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted_positive = probability >= threshold
        tp = int(np.sum((y == 1) & predicted_positive))
        fp = int(np.sum((y == 0) & predicted_positive))
        weight = threshold / (1 - threshold)
        model_nb = tp / len(y) - fp / len(y) * weight
        treat_all = prevalence - (1 - prevalence) * weight
        rows.append(
            {
                "Variant": variant,
                "Target": target,
                "Threshold_Probability": float(threshold),
                "Model_Net_Benefit": float(model_nb),
                "Treat_All_Net_Benefit": float(treat_all),
                "Treat_None_Net_Benefit": 0.0,
                "TP": tp,
                "FP": fp,
            }
        )
    table = pd.DataFrame(rows)
    useful = table["Model_Net_Benefit"] > np.maximum(table["Treat_All_Net_Benefit"], 0.0)
    thresholds_useful = table.loc[useful, "Threshold_Probability"]
    summary = {
        "Variant": variant,
        "Target": target,
        "Evaluated_Threshold_N": int(len(table)),
        "Clinically_Useful_Threshold_N": int(useful.sum()),
        "Clinically_Useful_Threshold_Fraction": float(useful.mean()),
        "Useful_Threshold_Min": float(thresholds_useful.min()) if len(thresholds_useful) else math.nan,
        "Useful_Threshold_Max": float(thresholds_useful.max()) if len(thresholds_useful) else math.nan,
        "Max_Model_Net_Benefit": float(table["Model_Net_Benefit"].max()),
    }
    return table, summary


def patient_cluster_bootstrap(
    frame: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Compare uncalibrated and Platt probabilities with patient resampling."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for target in TARGETS:
        group = frame.loc[frame["Target"] == target].copy()
        patients, codes = np.unique(group["Patient_ID"].astype(str), return_inverse=True)
        y = group["True_Label"].to_numpy(dtype=int)
        raw = group["Raw_Prob"].to_numpy(dtype=float)
        platt = group["Platt_Prob"].to_numpy(dtype=float)
        point_raw = discrimination_metrics(y, raw)
        point_platt = discrimination_metrics(y, platt)
        distributions: dict[str, list[float]] = {"AUROC": [], "AUPRC": [], "Brier_Score": []}
        valid = 0
        for _ in range(n_bootstrap):
            sampled = rng.integers(0, len(patients), len(patients))
            weights = np.bincount(sampled, minlength=len(patients))[codes]
            if np.sum(weights * y) == 0 or np.sum(weights * (1 - y)) == 0:
                continue
            boot_raw = discrimination_metrics(y, raw, weights)
            boot_platt = discrimination_metrics(y, platt, weights)
            for metric in distributions:
                distributions[metric].append(boot_platt[metric] - boot_raw[metric])
            valid += 1
        for metric, values in distributions.items():
            distribution = np.asarray(values, dtype=float)
            delta = point_platt[metric] - point_raw[metric]
            rows.append(
                {
                    "Target": target,
                    "Comparison": "Platt_calibrated_minus_original_uncalibrated",
                    "Metric": metric,
                    "Original_Value": point_raw[metric],
                    "Platt_Value": point_platt[metric],
                    "Delta": delta,
                    "CI_95_Low": float(np.percentile(distribution, 2.5)) if len(distribution) else math.nan,
                    "CI_95_High": float(np.percentile(distribution, 97.5)) if len(distribution) else math.nan,
                    "N_Patients": int(len(patients)),
                    "N_Nodes": int(len(group)),
                    "N_Bootstrap_Requested": n_bootstrap,
                    "N_Bootstrap_Valid": valid,
                    "Seed": seed,
                }
            )
    return pd.DataFrame(rows)


def model_manifest(model_paths: dict[str, Any] | None) -> dict[str, Any]:
    if not model_paths:
        return {}
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - depends on optional input
        raise RuntimeError("xgboost is required when model_paths are supplied") from exc
    output: dict[str, Any] = {}
    for label, value in model_paths.items():
        path = Path(str(value)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist for {label}: {path}")
        booster = xgb.Booster()
        booster.load_model(path)
        before = sha256(path)
        output[str(label)] = {
            "file_name": path.name,
            "sha256_before": before,
            "num_boosted_rounds": int(booster.num_boosted_rounds()),
            "sha256_after": sha256(path),
        }
        output[str(label)]["hash_unchanged"] = output[str(label)]["sha256_before"] == output[str(label)]["sha256_after"]
    return output


def ensure_external_output(output_dir: Path, module_root: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    module_root = module_root.resolve()
    try:
        output_dir.relative_to(module_root)
    except ValueError:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    raise ValueError("--output-dir must be outside the repository calibration module")
