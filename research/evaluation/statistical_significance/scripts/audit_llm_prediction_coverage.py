from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "evaluation" / "statistical_significance"
RESULTS_DIR = OUT_DIR / "results" / "statistical_audit"
DOCS_DIR = OUT_DIR / "docs"

TARGETS = ("1m", "1y", "5y")

MODELS = {
    "Qwen3-4B SFT": PROJECT_ROOT / "private_inputs" / "llm" / "qwen3_4b_sft" / "test_evaluation_log.jsonl",
    "Qwen3-4B baseline": PROJECT_ROOT / "private_inputs" / "llm" / "qwen3_4b" / "test_evaluation_log.jsonl",
    "Llama3.1-8B": PROJECT_ROOT / "private_inputs" / "llm" / "llama3.1_8b" / "test_evaluation_log.jsonl",
    "Huatuo-O1-7B": PROJECT_ROOT / "private_inputs" / "llm" / "huatuo_o1_7b" / "test_evaluation_log.jsonl",
}

DATA_ROOT = Path(os.environ.get("PEDILT_DATA_DIR", PROJECT_ROOT / "data"))
GOLD_TEST = DATA_ROOT / "test_dataset_gold.json"
XGB_SOURCE = PROJECT_ROOT / "machine_learning" / "train" / "xgboost" / "xgb_sequence_saved_test_predictions.csv"


def parse_prob(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if 0.0 <= number <= 1.0:
        return number
    return np.nan


def parse_label(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if number in (0, 1):
        return float(number)
    return np.nan


def read_gold_nodes() -> dict[str, dict[str, int]]:
    with GOLD_TEST.open("r", encoding="utf-8") as handle:
        nodes = json.load(handle)
    gold: dict[str, dict[str, int]] = {}
    for node in nodes:
        node_id = str(node.get("node_id") or node.get("id") or "")
        labels = node.get("true_labels") or node.get("真实标签") or {}
        if not node_id:
            continue
        gold[node_id] = {}
        for target in TARGETS:
            label = parse_label(labels.get(target))
            if np.isfinite(label):
                gold[node_id][target] = int(label)
    return gold


def read_xgb_reference_nodes() -> dict[str, dict[str, int]]:
    target_map = {"Label_1m": "1m", "Label_1y": "1y", "Label_5y": "5y"}
    reference: dict[str, dict[str, int]] = defaultdict(dict)
    with XGB_SOURCE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = target_map.get(row.get("Target", ""))
            if target is None:
                continue
            sample_id = str(row.get("Sample_ID", ""))
            label = parse_label(row.get("True_Label"))
            if sample_id and np.isfinite(label):
                reference[sample_id][target] = int(label)
    return dict(reference)


def read_model_records(path: Path) -> tuple[dict[str, dict[str, Any]], Counter]:
    latest: dict[str, dict[str, Any]] = {}
    status_counter: Counter = Counter()
    duplicate_counter: Counter = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                status_counter["jsonl_decode_failed"] += 1
                continue
            node_id = str(record.get("node_id", ""))
            if not node_id:
                status_counter["missing_node_id"] += 1
                continue
            if node_id in latest:
                duplicate_counter[node_id] += 1
            status = str(record.get("process_status", "missing_status") or "missing_status")
            status_counter[status] += 1
            record["_line_no"] = line_no
            latest[node_id] = record
    status_counter["unique_node_count"] = len(latest)
    status_counter["duplicate_node_count"] = sum(duplicate_counter.values())
    status_counter["duplicated_unique_node_count"] = len(duplicate_counter)
    return latest, status_counter


def record_target_status(record: dict[str, Any] | None, target: str) -> tuple[str, float, float]:
    if record is None:
        return "missing_record", np.nan, np.nan
    labels = record.get("true_labels") or {}
    probs = record.get("pred_probs") or record.get("injected_pred_probs") or {}
    label = parse_label(labels.get(target))
    prob = parse_prob(probs.get(target))
    process_status = str(record.get("process_status", "") or "")
    if process_status and process_status != "success":
        return process_status, label, prob
    if target not in labels:
        return "missing_label", label, prob
    if target not in probs:
        return "missing_pred_prob", label, prob
    if not np.isfinite(label):
        return "invalid_label", label, prob
    if not np.isfinite(prob):
        return "invalid_pred_prob", label, prob
    return "valid", label, prob


def auroc_or_nan(labels: list[int], probs: list[float]) -> float:
    if len(set(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, probs))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    gold_nodes = read_gold_nodes()
    xgb_nodes = read_xgb_reference_nodes()
    standard_nodes = xgb_nodes

    model_records: dict[str, dict[str, dict[str, Any]]] = {}
    status_rows: list[dict[str, Any]] = []
    for model, path in MODELS.items():
        records, status_counter = read_model_records(path)
        model_records[model] = records
        latest_counter = Counter(str(record.get("process_status", "missing_status") or "missing_status") for record in records.values())
        status_rows.append(
            {
                "Model": model,
                "Source": str(path.relative_to(PROJECT_ROOT)),
                "Log_Lines": sum(
                    count
                    for key, count in status_counter.items()
                    if key
                    not in {
                        "unique_node_count",
                        "duplicate_node_count",
                        "duplicated_unique_node_count",
                    }
                ),
                "Unique_Node_N": status_counter["unique_node_count"],
                "Duplicate_Record_N": status_counter["duplicate_node_count"],
                "Duplicated_Node_N": status_counter["duplicated_unique_node_count"],
                "Success_Record_N": status_counter["success"],
                "Parse_Failed_Record_N": status_counter["parse_failed"],
                "API_Failed_Record_N": status_counter["api_failed"],
                "Final_Unique_Success_Node_N": latest_counter["success"],
                "Final_Unique_Parse_Failed_Node_N": latest_counter["parse_failed"],
                "Final_Unique_API_Failed_Node_N": latest_counter["api_failed"],
                "Other_Status_N": sum(
                    count
                    for key, count in status_counter.items()
                    if key
                    not in {
                        "unique_node_count",
                        "duplicate_node_count",
                        "duplicated_unique_node_count",
                        "success",
                        "parse_failed",
                        "api_failed",
                    }
                ),
            }
        )

    write_csv(RESULTS_DIR / "llm_log_record_status_summary.csv", status_rows)

    coverage_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    standalone_rows: list[dict[str, Any]] = []
    xgb_expected_by_target = {target: {nid for nid, labels in standard_nodes.items() if target in labels} for target in TARGETS}

    valid_sets: dict[tuple[str, str], set[str]] = {}
    y_by_model_target: dict[tuple[str, str], dict[str, int]] = {}
    p_by_model_target: dict[tuple[str, str], dict[str, float]] = {}

    for model, records in model_records.items():
        for target in TARGETS:
            expected_ids = xgb_expected_by_target[target]
            status_count: Counter = Counter()
            labels_map: dict[str, int] = {}
            probs_map: dict[str, float] = {}
            for node_id in sorted(expected_ids):
                status, label, prob = record_target_status(records.get(node_id), target)
                status_count[status] += 1
                if status == "valid":
                    labels_map[node_id] = int(label)
                    probs_map[node_id] = float(prob)
                else:
                    invalid_rows.append(
                        {
                            "Model": model,
                            "Target": target,
                            "Sample_ID": node_id,
                            "Status": status,
                            "Has_Record": node_id in records,
                            "Process_Status": records.get(node_id, {}).get("process_status") if node_id in records else "",
                            "Line_No": records.get(node_id, {}).get("_line_no") if node_id in records else "",
                        }
                    )

            valid_ids = set(labels_map)
            valid_sets[(model, target)] = valid_ids
            y_by_model_target[(model, target)] = labels_map
            p_by_model_target[(model, target)] = probs_map
            labels = [labels_map[nid] for nid in sorted(valid_ids)]
            probs = [probs_map[nid] for nid in sorted(valid_ids)]

            standalone_rows.append(
                {
                    "Model": model,
                    "Target": target,
                    "Valid_N": len(valid_ids),
                    "Positive_N": int(sum(labels)),
                    "Negative_N": int(len(labels) - sum(labels)),
                    "AUROC": auroc_or_nan(labels, probs),
                }
            )
            coverage_rows.append(
                {
                    "Model": model,
                    "Target": target,
                    "Expected_N_XGB_Reference": len(expected_ids),
                    "Valid_N": status_count["valid"],
                    "Missing_Record_N": status_count["missing_record"],
                    "API_Failed_N": status_count["api_failed"],
                    "Parse_Failed_N": status_count["parse_failed"],
                    "Missing_Pred_Prob_N": status_count["missing_pred_prob"],
                    "Invalid_Pred_Prob_N": status_count["invalid_pred_prob"],
                    "Missing_Label_N": status_count["missing_label"],
                    "Invalid_Label_N": status_count["invalid_label"],
                    "Other_Invalid_N": sum(
                        count
                        for status, count in status_count.items()
                        if status
                        not in {
                            "valid",
                            "missing_record",
                            "api_failed",
                            "parse_failed",
                            "missing_pred_prob",
                            "invalid_pred_prob",
                            "missing_label",
                            "invalid_label",
                        }
                    ),
                    "Recoverable_By_Retry_N": status_count["missing_record"]
                    + status_count["api_failed"]
                    + status_count["parse_failed"]
                    + status_count["missing_pred_prob"]
                    + status_count["invalid_pred_prob"],
                }
            )

            list_rows = [
                row
                for row in invalid_rows
                if row["Model"] == model
                and row["Target"] == target
                and row["Status"]
                in {
                    "missing_record",
                    "api_failed",
                    "parse_failed",
                    "missing_pred_prob",
                    "invalid_pred_prob",
                }
            ]
            filename_model = (
                model.lower()
                .replace(" ", "_")
                .replace(".", "")
                .replace("-", "_")
                .replace("/", "_")
            )
            write_csv(RESULTS_DIR / f"retry_candidates_{filename_model}_{target}.csv", list_rows)

    write_csv(RESULTS_DIR / "llm_target_coverage_summary.csv", coverage_rows)
    write_csv(RESULTS_DIR / "llm_standalone_auroc_from_current_logs.csv", standalone_rows)
    write_csv(RESULTS_DIR / "llm_invalid_or_missing_target_records.csv", invalid_rows)

    common_rows: list[dict[str, Any]] = []
    all_models = list(MODELS)
    for target in TARGETS:
        expected_ids = xgb_expected_by_target[target]
        common_ids = set(expected_ids)
        for model in all_models:
            common_ids &= valid_sets[(model, target)]
        for model in all_models:
            ids = sorted(common_ids)
            labels = [y_by_model_target[(model, target)][nid] for nid in ids]
            probs = [p_by_model_target[(model, target)][nid] for nid in ids]
            common_rows.append(
                {
                    "Scope": "All_4_LLM_Common_Valid",
                    "Model": model,
                    "Target": target,
                    "N": len(ids),
                    "Positive_N": int(sum(labels)),
                    "Negative_N": int(len(labels) - sum(labels)),
                    "AUROC": auroc_or_nan(labels, probs),
                }
            )

    write_csv(RESULTS_DIR / "llm_common_valid_auroc_all4.csv", common_rows)

    pairwise_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for i, model_a in enumerate(all_models):
            for model_b in all_models[i + 1 :]:
                ids = sorted(valid_sets[(model_a, target)] & valid_sets[(model_b, target)])
                labels_a = [y_by_model_target[(model_a, target)][nid] for nid in ids]
                labels_b = [y_by_model_target[(model_b, target)][nid] for nid in ids]
                if labels_a != labels_b:
                    matched = [
                        nid
                        for nid in ids
                        if y_by_model_target[(model_a, target)][nid] == y_by_model_target[(model_b, target)][nid]
                    ]
                    ids = matched
                    labels_a = [y_by_model_target[(model_a, target)][nid] for nid in ids]
                probs_a = [p_by_model_target[(model_a, target)][nid] for nid in ids]
                probs_b = [p_by_model_target[(model_b, target)][nid] for nid in ids]
                pairwise_rows.append(
                    {
                        "Target": target,
                        "Model_A": model_a,
                        "Model_B": model_b,
                        "Paired_Common_N": len(ids),
                        "Positive_N": int(sum(labels_a)),
                        "Negative_N": int(len(labels_a) - sum(labels_a)),
                        "AUROC_A": auroc_or_nan(labels_a, probs_a),
                        "AUROC_B": auroc_or_nan(labels_a, probs_b),
                        "AUROC_Delta_A_minus_B": auroc_or_nan(labels_a, probs_a)
                        - auroc_or_nan(labels_a, probs_b),
                    }
                )
    write_csv(RESULTS_DIR / "llm_pairwise_common_valid_auroc.csv", pairwise_rows)

    target_union_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        expected_ids = xgb_expected_by_target[target]
        valid_by_all_or_missing = {
            node_id
            for node_id in expected_ids
            if all(node_id in valid_sets[(model, target)] for model in all_models)
        }
        target_union_rows.append(
            {
                "Target": target,
                "XGB_Reference_N": len(expected_ids),
                "All_4_LLM_Common_Valid_N": len(valid_by_all_or_missing),
                "Need_At_Least_One_LLM_Retry_N": len(expected_ids - valid_by_all_or_missing),
            }
        )
    write_csv(RESULTS_DIR / "llm_common_sample_alignment_summary.csv", target_union_rows)

    report = [
        "# LLM Prediction Coverage Audit",
        "",
        "## Purpose",
        "",
        "This audit checks whether current LLM AUROC values and pairwise statistical comparisons use the same effective samples. Raw clinical prompts and raw model responses are not exported.",
        "",
        "## Source Files",
        "",
    ]
    for model, path in MODELS.items():
        report.append(f"- {model}: `{path.relative_to(PROJECT_ROOT)}`")
    report.extend(
        [
            f"- Evaluation test set: `{GOLD_TEST.relative_to(PROJECT_ROOT)}`",
            f"- Reference complete target sample set: `{XGB_SOURCE.relative_to(PROJECT_ROOT)}`",
            "",
            "## Generated Tables",
            "",
            "- `llm_log_record_status_summary.csv`: log line, success, failure, and duplicate summary.",
            "- `llm_target_coverage_summary.csv`: valid/missing/invalid counts by model and endpoint.",
            "- `llm_standalone_auroc_from_current_logs.csv`: AUROC using each model's current valid samples.",
            "- `llm_common_valid_auroc_all4.csv`: AUROC after restricting all four LLMs to the same valid samples.",
            "- `llm_pairwise_common_valid_auroc.csv`: AUROC on pairwise common valid samples.",
            "- `retry_candidates_<model>_<target>.csv`: node IDs eligible for an inference retry or rescue parsing.",
            "",
            "## Interpretation Rule",
            "",
            "For reporting consistency, standalone model-performance tables and statistical-difference tables should either both use each model's full valid set with explicit N reporting, or both use a pre-specified common valid test set. Inference may be retried only for records with missing predictions, API failures, parsing failures, or invalid probabilities, using the same frozen prompt and model configuration.",
        ]
    )
    (DOCS_DIR / "llm_prediction_coverage_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
