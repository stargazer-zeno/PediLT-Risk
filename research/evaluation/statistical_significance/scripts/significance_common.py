"""Shared utilities for patient-cluster bootstrap significance analyses."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = PROJECT_ROOT / "evaluation" / "statistical_significance"
RESULTS_DIR = ANALYSIS_ROOT / "results"
FIGURES_DIR = ANALYSIS_ROOT / "figures"
DOCS_DIR = ANALYSIS_ROOT / "docs"
SCRIPTS_DIR = ANALYSIS_ROOT / "scripts"
FINAL_COMMON_PREDICTIONS = Path(
    os.environ.get(
        "PEDILT_FINAL_COMMON_PREDICTIONS",
        PROJECT_ROOT / "private_inputs" / "final_common_cohort_predictions_long.csv",
    )
)

TARGETS = ("1m", "1y", "5y")
TARGET_TO_LABEL = {"1m": "Label_1m", "1y": "Label_1y", "5y": "Label_5y"}
LABEL_TO_TARGET = {v: k for k, v in TARGET_TO_LABEL.items()}

MODEL_ORDER = (
    "XGBoost",
    "LSTM",
    "RSF",
    "Qwen3-4B SFT",
    "Qwen3-4B baseline",
    "Llama3.1-8B",
    "Huatuo-O1-7B",
)

STAGE_ORDER = ("Stage1", "Stage2", "Stage3", "Stage4", "Stage5")
STAGE_DISPLAY = {
    "Stage1": "0-1 mo",
    "Stage2": "2-3 mo",
    "Stage3": "4-12 mo",
    "Stage4": "1-2 y",
    "Stage5": ">2 y",
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    family: str
    kind: str
    path: str


MODEL_CONFIGS = (
    ModelConfig("XGBoost", "ML", "ml_csv", "machine_learning/train/xgboost/xgb_sequence_saved_test_predictions.csv"),
    ModelConfig("LSTM", "ML", "ml_csv", "machine_learning/train/lstm/lstm_sequence_saved_test_predictions.csv"),
    ModelConfig("RSF", "ML", "ml_csv", "machine_learning/train/rsf/rsf_sequence_saved_test_predictions.csv"),
    ModelConfig("Qwen3-4B SFT", "LLM", "llm_jsonl", "private_inputs/llm/qwen3_4b_sft/test_evaluation_log.jsonl"),
    ModelConfig("Qwen3-4B baseline", "LLM", "llm_jsonl", "private_inputs/llm/qwen3_4b/test_evaluation_log.jsonl"),
    ModelConfig("Llama3.1-8B", "LLM", "llm_jsonl", "private_inputs/llm/llama3.1_8b/test_evaluation_log.jsonl"),
    ModelConfig("Huatuo-O1-7B", "LLM", "llm_jsonl", "private_inputs/llm/huatuo_o1_7b/test_evaluation_log.jsonl"),
)


def ensure_output_dirs() -> None:
    for path in (ANALYSIS_ROOT, RESULTS_DIR, FIGURES_DIR, DOCS_DIR, SCRIPTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def standardize_target(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in TARGETS:
        return text
    return LABEL_TO_TARGET.get(text)


def parse_patient_id(sample_id: object) -> str:
    text = str(sample_id)
    if "_node_" in text:
        return text.split("_node_", 1)[0]
    return text.split("_", 1)[0]


def parse_visit_count(sample_id: object) -> float:
    text = str(sample_id)
    if "_node_" not in text:
        return np.nan
    try:
        return float(text.split("_node_", 1)[1])
    except ValueError:
        return np.nan


def parse_label(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "是", "死亡"}:
        return 1.0
    if text in {"0", "0.0", "false", "no", "n", "否", "未死亡", "存活"}:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        return np.nan
    if number in (0.0, 1.0):
        return number
    return np.nan


def load_ml_predictions(config: ModelConfig) -> pd.DataFrame:
    path = PROJECT_ROOT / config.path
    df = pd.read_csv(path)
    df["Target"] = df["Target"].map(standardize_target)
    df["True_Label"] = df["True_Label"].map(parse_label)
    df["Pred_Prob"] = pd.to_numeric(df["Pred_Prob"], errors="coerce")
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    df["Sample_ID"] = df["Sample_ID"].astype(str)
    df["Visit_Count"] = pd.to_numeric(df["Visit_Count"], errors="coerce")
    df["Model"] = config.name
    df["Model_Family"] = config.family
    return df[["Model", "Model_Family", "Target", "Sample_ID", "Patient_ID", "Visit_Count", "True_Label", "Pred_Prob"]]


def load_llm_predictions(config: ModelConfig) -> pd.DataFrame:
    path = PROJECT_ROOT / config.path
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("node_id", ""))
            if not sample_id:
                continue
            labels = record.get("true_labels") or {}
            probs = record.get("pred_probs") or record.get("injected_pred_probs") or {}
            for target in TARGETS:
                if target not in labels or target not in probs:
                    continue
                rows.append(
                    {
                        "Model": config.name,
                        "Model_Family": config.family,
                        "Target": target,
                        "Sample_ID": sample_id,
                        "Patient_ID": parse_patient_id(sample_id),
                        "Visit_Count": parse_visit_count(sample_id),
                        "True_Label": parse_label(labels.get(target)),
                        "Pred_Prob": pd.to_numeric(probs.get(target), errors="coerce"),
                    }
                )
    return pd.DataFrame.from_records(rows)


def load_final_common_predictions() -> pd.DataFrame:
    if not FINAL_COMMON_PREDICTIONS.exists():
        raise FileNotFoundError(FINAL_COMMON_PREDICTIONS)
    df = pd.read_csv(FINAL_COMMON_PREDICTIONS)
    required = {
        "Model",
        "Model_Family",
        "Target",
        "Sample_ID",
        "Patient_ID",
        "Visit_Count",
        "True_Label",
        "Pred_Prob",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {FINAL_COMMON_PREDICTIONS}: {sorted(missing)}")
    df = df[list(required) + ([col for col in ("Stage", "Stage_Label", "Day_Range") if col in df.columns])].copy()
    df["Target"] = df["Target"].map(standardize_target)
    df["Sample_ID"] = df["Sample_ID"].astype(str)
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    df["Visit_Count"] = pd.to_numeric(df["Visit_Count"], errors="coerce")
    df["True_Label"] = df["True_Label"].map(parse_label)
    df["Pred_Prob"] = pd.to_numeric(df["Pred_Prob"], errors="coerce")
    df = df[
        df["Model"].isin(MODEL_ORDER)
        & df["Target"].isin(TARGETS)
        & np.isfinite(df["True_Label"])
        & np.isfinite(df["Pred_Prob"])
        & (df["Pred_Prob"] >= 0.0)
        & (df["Pred_Prob"] <= 1.0)
    ].copy()
    df["True_Label"] = df["True_Label"].astype(int)
    return df[["Model", "Model_Family", "Target", "Sample_ID", "Patient_ID", "Visit_Count", "True_Label", "Pred_Prob"] + ([col for col in ("Stage", "Stage_Label", "Day_Range") if col in df.columns])]


def load_predictions(configs: Iterable[ModelConfig] = MODEL_CONFIGS) -> pd.DataFrame:
    if configs is MODEL_CONFIGS:
        return load_final_common_predictions()
    frames = []
    for config in configs:
        if config.kind == "ml_csv":
            frames.append(load_ml_predictions(config))
        elif config.kind == "llm_jsonl":
            frames.append(load_llm_predictions(config))
        else:
            raise ValueError(f"Unsupported model kind: {config.kind}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Target"].isin(TARGETS)].copy()
    df = df[
        np.isfinite(df["True_Label"])
        & np.isfinite(df["Pred_Prob"])
        & (df["Pred_Prob"] >= 0.0)
        & (df["Pred_Prob"] <= 1.0)
    ].copy()
    df["True_Label"] = df["True_Label"].astype(int)
    df["Pred_Prob"] = df["Pred_Prob"].clip(0.0, 1.0)
    return df


def load_stage_map() -> dict[str, str]:
    path = Path(os.environ.get("PEDILT_STAGE_MAP", PROJECT_ROOT / "private_inputs" / "test_ids_by_stage.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    stage_map: dict[str, str] = {}
    for raw_stage, sample_ids in data.items():
        stage = raw_stage[:1].upper() + raw_stage[1:] if raw_stage.startswith("stage") else raw_stage
        for sample_id in sample_ids:
            stage_map[str(sample_id)] = stage
    return stage_map


def add_stage(predictions: pd.DataFrame) -> pd.DataFrame:
    if "Stage" in predictions.columns:
        return predictions.copy()
    stage_map = load_stage_map()
    df = predictions.copy()
    df["Stage"] = df["Sample_ID"].map(stage_map)
    return df


def model_pairs_all() -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(MODEL_ORDER) for b in MODEL_ORDER[i + 1 :]]


def get_paired_predictions(
    predictions: pd.DataFrame,
    model_a: str,
    model_b: str,
    target: str,
    stage: str | None = None,
) -> pd.DataFrame:
    keys = ["Target", "Sample_ID"]
    df = predictions
    if stage is not None:
        df = df[df["Stage"] == stage]
    a = df[(df["Model"] == model_a) & (df["Target"] == target)].copy()
    b = df[(df["Model"] == model_b) & (df["Target"] == target)].copy()
    a = a.rename(
        columns={
            "Patient_ID": "Patient_ID_A",
            "Visit_Count": "Visit_Count_A",
            "True_Label": "True_Label_A",
            "Pred_Prob": "Pred_Prob_A",
        }
    )
    b = b.rename(
        columns={
            "Patient_ID": "Patient_ID_B",
            "Visit_Count": "Visit_Count_B",
            "True_Label": "True_Label_B",
            "Pred_Prob": "Pred_Prob_B",
        }
    )
    merged = a[keys + ["Patient_ID_A", "Visit_Count_A", "True_Label_A", "Pred_Prob_A"]].merge(
        b[keys + ["Patient_ID_B", "Visit_Count_B", "True_Label_B", "Pred_Prob_B"]],
        on=keys,
        how="inner",
    )
    merged = merged[merged["True_Label_A"] == merged["True_Label_B"]].copy()
    merged["Patient_ID"] = merged["Patient_ID_A"].where(merged["Patient_ID_A"].notna(), merged["Patient_ID_B"])
    merged["Visit_Count"] = merged["Visit_Count_A"].where(merged["Visit_Count_A"].notna(), merged["Visit_Count_B"])
    merged["True_Label"] = merged["True_Label_A"].astype(int)
    if stage is not None:
        merged["Stage"] = stage
    return merged[
        [
            "Target",
            "Sample_ID",
            "Patient_ID",
            "Visit_Count",
            "True_Label",
            "Pred_Prob_A",
            "Pred_Prob_B",
        ]
        + (["Stage"] if stage is not None else [])
    ].reset_index(drop=True)


def patient_codes(patient_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    unique, codes = np.unique(patient_ids.astype(str).to_numpy(), return_inverse=True)
    return codes.astype(int), unique


def auc_or_nan(y_true: np.ndarray, score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    mask = np.isfinite(y_true) & np.isfinite(score)
    if sample_weight is not None:
        mask &= np.isfinite(sample_weight) & (sample_weight > 0)
    if mask.sum() == 0:
        return np.nan
    y = y_true[mask].astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    if sample_weight is None:
        return float(roc_auc_score(y, score[mask]))
    return float(roc_auc_score(y, score[mask], sample_weight=sample_weight[mask]))


def weighted_auc_from_ranks(y_true: np.ndarray, score: np.ndarray, weights: np.ndarray) -> np.ndarray:
    y = y_true.astype(int)
    ranks = rankdata(score, method="average").astype(float)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    pos_mask = y == 1
    neg_mask = y == 0
    pos_w = weights[:, pos_mask]
    neg_w = weights[:, neg_mask]
    pos_total = pos_w.sum(axis=1)
    neg_total = neg_w.sum(axis=1)
    rank_sum_pos = (pos_w * ranks[pos_mask]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (rank_sum_pos - pos_total * (pos_total + 1.0) / 2.0) / (pos_total * neg_total)
    auc[(pos_total <= 0) | (neg_total <= 0)] = np.nan
    return auc


def bootstrap_auc_deltas_fast(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    patient_ids: pd.Series,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 1000,
) -> np.ndarray:
    codes, unique_patients = patient_codes(patient_ids)
    n_patients = len(unique_patients)
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, n_bootstrap, batch_size):
        size = min(batch_size, n_bootstrap - start)
        sampled = rng.integers(0, n_patients, size=(size, n_patients))
        patient_weights = np.zeros((size, n_patients), dtype=float)
        row_idx = np.repeat(np.arange(size), n_patients)
        np.add.at(patient_weights, (row_idx, sampled.ravel()), 1.0)
        node_weights = patient_weights[:, codes]
        auc_a = weighted_auc_from_ranks(y_true, score_a, node_weights)
        auc_b = weighted_auc_from_ranks(y_true, score_b, node_weights)
        delta = auc_a - auc_b
        chunks.append(delta[np.isfinite(delta)])
    if not chunks:
        return np.asarray([], dtype=float)
    return np.concatenate(chunks)


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    finite_idx = np.where(finite)[0]
    finite_p = p[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q[finite_idx[order]] = adjusted
    return q


def significance_stars(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def make_model_inventory(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target), group in predictions.groupby(["Model", "Target"], sort=False):
        positives = int(group["True_Label"].sum())
        rows.append(
            {
                "Model": model,
                "Target": target,
                "N_Nodes": int(len(group)),
                "N_Patients": int(group["Patient_ID"].nunique()),
                "Positive_N": positives,
                "Negative_N": int(len(group) - positives),
                "AUROC": auc_or_nan(group["True_Label"].to_numpy(), group["Pred_Prob"].to_numpy()),
                "Prediction_Source": str(FINAL_COMMON_PREDICTIONS.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)
