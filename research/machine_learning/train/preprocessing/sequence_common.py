import json
import math
import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = REPO_ROOT / "machine_learning" / "train"
DATASET_DIR = TRAIN_ROOT / "datasets"

DATA_ROOT = Path(os.environ.get("PEDILT_DATA_DIR", REPO_ROOT / "data"))
TRAIN_JSON = DATA_ROOT / "train_dataset_gold.json"
TEST_JSON = DATA_ROOT / "test_dataset_gold.json"

TARGETS = ["Label_1m", "Label_1y", "Label_5y"]
HORIZON_KEYS = ["1m", "1y", "5y"]
TIME_POINTS = {"Label_1m": 30.0, "Label_1y": 365.0, "Label_5y": 1825.0}

OUTCOME_BASE_KEYS = {"是否死亡", "死亡时间", "死亡原因", "随访记录摘要"}
CONTINUOUS_BASE_COLS = ["手术年龄", "患儿术时体重", "供肝重量", "GRWR", "供体年龄"]
KNOWN_CATEGORICAL_BASE_COLS = [
    "术式",
    "患儿血型",
    "供体血型",
    "ABO血型相容性",
    "主要诊断分类",
    "手术史",
    "患儿CYP",
    "供体CYP",
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def patient_id(sample_id):
    sample_id = str(sample_id)
    return sample_id.split("_node_")[0] if "_node_" in sample_id else sample_id


def visit_count(sample_id):
    sample_id = str(sample_id)
    if "_node_" not in sample_id:
        return 1
    try:
        return int(sample_id.rsplit("_node_", 1)[1])
    except Exception:
        return 1


def clean_num(value):
    value = str(value).strip()
    if value in {"", "NaN", "None", "nan"}:
        return np.nan
    if "阴" in value or "(-)" in value or "（-）" in value or "（—）" in value:
        return 0.0
    if "阳" in value or "(+)" in value or "（+）" in value:
        return 1.0
    if "/" in value:
        parts = [p for p in value.split("/") if p.strip()]
        if parts:
            value = parts[0]
    value = value.replace("E*7", "e7").replace("E*", "e")
    match = re.search(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", value)
    if not match:
        return np.nan
    try:
        number = float(match.group())
    except Exception:
        return np.nan
    if number > 1e10 or number < -1000:
        return np.nan
    return number


def split_series_field(field):
    text = str(field).replace("。", "")
    parts = text.split(": ", 1)
    if len(parts) != 2:
        parts = text.split(":", 1)
    if len(parts) != 2:
        return None, []
    name = parts[0].strip()
    values = [v.strip() for v in parts[1].split(", ")]
    return name, values


def parse_date(value):
    if value is None:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(), "%Y-%m-%d")
    except Exception:
        return None


def current_followup_date(item):
    summary = item.get("基础信息", {}).get("随访记录摘要", "")
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", str(summary))
    if dates:
        return parse_date(dates[-1])

    surgery = parse_date(item.get("基础信息", {}).get("手术日期"))
    if surgery is None:
        return None

    postop = None
    for field in item.get("时序随访基础信息", []):
        name, values = split_series_field(field)
        if name != "术后天数":
            continue
        for value in reversed(values):
            try:
                postop = float(value)
                break
            except Exception:
                continue
    if postop is None:
        return surgery
    return surgery + timedelta(days=int(round(postop)))


def fallback_time_from_labels(labels):
    if labels.get("1m") == 1:
        return True, 15.0
    if labels.get("1y") == 1:
        return True, 180.0
    if labels.get("5y") == 1:
        return True, 1000.0
    if labels.get("5y") == 0:
        return False, 1825.0
    if labels.get("1y") == 0:
        return False, 365.0
    return False, 30.0


def survival_lookup_from_items(items):
    lookup = {}
    for item in items:
        sid = str(item.get("id", ""))
        labels = item.get("真实标签", {})
        current = current_followup_date(item)
        death = parse_date(item.get("基础信息", {}).get("死亡时间"))
        is_dead = str(labels.get("是否死亡")).lower() in {"1", "true"}
        if is_dead and current is not None and death is not None and death > current:
            event = True
            time = max((death - current).days, 1)
        else:
            event, time = fallback_time_from_labels(labels)
        lookup[sid] = (event, float(max(time, 1.0)))
    return lookup


def label_to_float(value):
    if value is None:
        return np.nan
    try:
        if isinstance(value, float) and math.isnan(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def score_binary(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
    return {
        "n": int(len(y_true)),
        "positive_n": int(np.sum(y_true == 1)),
        "negative_n": int(np.sum(y_true == 0)),
        "auroc": auroc,
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(data), f, ensure_ascii=False, indent=2)


def load_sequence_npz(split):
    return np.load(DATASET_DIR / f"sequence_{split}.npz", allow_pickle=False)


def load_schema():
    with open(DATASET_DIR / "schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


def format_metric(value):
    if value is None:
        return "nan"
    try:
        if not np.isfinite(value):
            return "nan"
    except Exception:
        return "nan"
    return f"{float(value):.4f}"
