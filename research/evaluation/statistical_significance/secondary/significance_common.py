from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = PROJECT_ROOT / "evaluation" / "statistical_significance" / "secondary_results"
RESULTS_DIR = ANALYSIS_ROOT / "results"
FIGURES_DIR = ANALYSIS_ROOT / "figures"
DOCS_DIR = ANALYSIS_ROOT / "docs"

TARGETS = ("1m", "1y", "5y")
TARGET_TO_LABEL = {"1m": "Label_1m", "1y": "Label_1y", "5y": "Label_5y"}
LABEL_TO_TARGET = {v: k for k, v in TARGET_TO_LABEL.items()}

METRICS = ("auroc", "brier")


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

PRIMARY_PAIRS = (
    ("XGBoost", "LSTM"),
    ("XGBoost", "RSF"),
    ("XGBoost", "Qwen3-4B SFT"),
    ("Qwen3-4B SFT", "Qwen3-4B baseline"),
    ("Qwen3-4B SFT", "Llama3.1-8B"),
    ("Qwen3-4B SFT", "Huatuo-O1-7B"),
)


def ensure_output_dirs() -> None:
    for path in (RESULTS_DIR, FIGURES_DIR, DOCS_DIR, ANALYSIS_ROOT / "scripts"):
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
    suffix = text.split("_node_", 1)[1]
    try:
        return float(suffix)
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
    if number == 0.0:
        return 0.0
    if number == 1.0:
        return 1.0
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


def load_predictions(configs: Iterable[ModelConfig] = MODEL_CONFIGS) -> pd.DataFrame:
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


def make_inventory(predictions: pd.DataFrame) -> pd.DataFrame:
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
                "Min_Pred_Prob": float(group["Pred_Prob"].min()),
                "Max_Pred_Prob": float(group["Pred_Prob"].max()),
            }
        )
    return pd.DataFrame(rows)


def get_model_pairs(mode: str) -> list[tuple[str, str]]:
    names = [config.name for config in MODEL_CONFIGS]
    if mode == "primary":
        return list(PRIMARY_PAIRS)
    if mode == "all":
        return [(a, b) for idx, a in enumerate(names) for b in names[idx + 1 :]]
    raise ValueError("pairs mode must be 'primary' or 'all'")


def get_paired_predictions(predictions: pd.DataFrame, model_a: str, model_b: str, target: str) -> pd.DataFrame:
    keys = ["Target", "Sample_ID"]
    a = predictions[(predictions["Model"] == model_a) & (predictions["Target"] == target)].copy()
    b = predictions[(predictions["Model"] == model_b) & (predictions["Target"] == target)].copy()
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
    ].reset_index(drop=True)


def paired_inventory(predictions: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for model_a, model_b in pairs:
        for target in TARGETS:
            paired = get_paired_predictions(predictions, model_a, model_b, target)
            positives = int(paired["True_Label"].sum()) if not paired.empty else 0
            rows.append(
                {
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "Target": target,
                    "N_Nodes": int(len(paired)),
                    "N_Patients": int(paired["Patient_ID"].nunique()) if not paired.empty else 0,
                    "Positive_N": positives,
                    "Negative_N": int(len(paired) - positives),
                }
            )
    return pd.DataFrame(rows)


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(q, index=p_values.index)
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(adjusted, 1.0)
    return pd.Series(q, index=p_values.index)


class WeightedMetricComputer:
    def __init__(self, y_true: np.ndarray, scores: np.ndarray):
        self.y = y_true.astype(float)
        self.scores = scores.astype(float)
        self.err2 = (self.scores - self.y) ** 2
        self._init_group_orders()

    def _init_group_orders(self) -> None:
        asc = np.argsort(self.scores, kind="mergesort")
        self.auc_y = self.y[asc]
        asc_scores = self.scores[asc]
        self.auc_group_starts = np.r_[0, np.flatnonzero(np.diff(asc_scores)) + 1]
        self.auc_order = asc

    def metric(self, metric_name: str, weights: np.ndarray) -> float:
        if metric_name == "auroc":
            return self.auroc(weights)
        if metric_name == "brier":
            return self.brier(weights)
        raise ValueError(f"Unsupported metric: {metric_name}")

    def auroc(self, weights: np.ndarray) -> float:
        w = weights[self.auc_order]
        pos_w = w * self.auc_y
        neg_w = w * (1.0 - self.auc_y)
        total_pos = pos_w.sum()
        total_neg = neg_w.sum()
        if total_pos <= 0 or total_neg <= 0:
            return np.nan
        group_pos = np.add.reduceat(pos_w, self.auc_group_starts)
        group_neg = np.add.reduceat(neg_w, self.auc_group_starts)
        neg_before = np.cumsum(group_neg) - group_neg
        numerator = np.sum(group_pos * (neg_before + 0.5 * group_neg))
        return float(numerator / (total_pos * total_neg))

    def brier(self, weights: np.ndarray) -> float:
        total = weights.sum()
        if total <= 0:
            return np.nan
        return float(np.sum(weights * self.err2) / total)


def patient_codes(patient_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    codes, uniques = pd.factorize(patient_ids.astype(str), sort=True)
    return codes.astype(int), uniques.astype(str)


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"
