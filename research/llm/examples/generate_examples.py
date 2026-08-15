"""Generate fully synthetic public examples using the project preprocessing logic."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))
PREPROCESS_DIR = RESEARCH_ROOT / "machine_learning" / "train" / "preprocessing"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from build_sequence_datasets import encode_static, parse_temporal
from llm.sft.build_sft_dataset import SYSTEM_PROMPT, build_final_prompt


LAB_VALUES = {
    "ALB": "32.0, 34.5, 36.8",
    "ALP": "280, 250, 220",
    "ALT": "135, 82, 49",
    "AST": "121, 74, 42",
    "CMV-DNA": "NaN, <400, <400",
    "CR": "35, 34, 32",
    "DB": "21.2, 14.8, 8.5",
    "EBV-DNA": "NaN, <400, <400",
    "HB": "93, 101, 108",
    "HBV-DNA": "NaN, <20, <20",
    "HBcAb": "阳性, 阳性, 阳性",
    "HBeAb": "阳性, 阳性, 阳性",
    "HBeAg": "阴性, 阴性, 阴性",
    "HBsAb": "23.4, 26.0, 30.5",
    "HBsAg": "阴性, 阴性, 阴性",
    "INR": "1.28, 1.15, 1.04",
    "N(%)": "62.0, 54.0, 47.0",
    "NT-proBNP": "230, 160, 92",
    "PLT": "102, 146, 185",
    "PT": "14.1, 12.8, 11.6",
    "TB": "47.0, 31.0, 20.0",
    "TP": "55.0, 60.0, 64.0",
    "WBC": "13.2, 10.1, 8.4",
    "γ-GT": "186, 104, 62",
    "免疫抑制剂浓度": "10.5, 8.6, 7.3",
    "嗜酸性粒细胞百分比": "0.01, 0.02, 0.03",
    "尿酸": "186, 194, 205",
    "总胆固醇": "3.8, 4.2, 4.5",
    "淋巴细胞绝对值": "1.8, 2.5, 3.1",
    "甘油三脂": "1.3, 1.1, 0.9",
    "甲胎蛋白": "35, 19, 8",
    "胆汁酸": "32, 19, 8",
    "血氨": "43, 31, 22",
    "血糖": "4.7, 4.9, 5.1",
    "雷帕浓度": "NaN, NaN, NaN",
}

MEDICATIONS = [
    "(乙10%)利伐沙班片(拜瑞妥)", "(乙10%)左乙拉西坦片", "(乙10%)恩替卡韦分散片",
    "(乙20%)丁二磺酸腺苷蛋氨酸片(思美泰)", "(甲)利可君片", "(甲)别嘌醇片",
    "(甲)复方磺胺甲噁唑片", "(甲)多烯磷脂酰胆碱胶囊(易善复)", "(甲)左甲状腺素钠片(优甲乐)",
    "(甲)托拉塞米片（特苏敏）", "(甲)氯硝西泮片", "(甲)甲钴胺片(弥可保)", "(甲)阿德福韦酯片",
    "万赛维", "他克莫司(普乐可复)", "他克莫司(赛福开)", "他克莫司缓释胶囊", "优思弗", "利可君",
    "华法林", "吗替麦考酚酯", "呋塞米片", "塞可平", "复方磺胺甲恶唑片", "天晴甘平",
    "奥美拉唑肠溶胶囊", "富马酸替诺福韦", "富马酸替诺福韦二吡呋酯片", "左乙拉西坦", "开浦兰",
    "恩替卡韦", "拉米夫定", "新山地明", "易善复", "更昔洛韦", "替格瑞洛片", "枸橼酸西地那非",
    "氟康唑胶囊", "环孢素", "甲泼尼龙片", "碳酸氢钠片", "米芙", "美卓乐", "美能", "螺内酯片",
    "赛可平", "醋酸泼尼松", "阿司匹林", "阿昔洛韦", "雷帕鸣", "骁悉",
]


def synthetic_patient() -> dict:
    medications = [f"{name}: NaN, 使用, 使用" if index % 3 == 0 else f"{name}: NaN, NaN, NaN" for index, name in enumerate(MEDICATIONS)]
    return {
        "id": "SYNTHETIC_PATIENT_001_node_3",
        "基础信息": {
            "患儿性别": "男", "供体性别": "女", "手术年龄": "1.2", "患儿术时体重": "9.6",
            "供肝重量": "280", "GRWR": "2.9", "供体年龄": "31", "术式": "活体",
            "患儿血型": "O", "供体血型": "O", "ABO血型相容性": "相同", "主要诊断分类": "胆汁淤积类疾病",
            "手术史": "葛西术后", "患儿CYP": "1/3", "供体CYP": "1/1", "手术日期": "2024-01-15",
            "随访记录摘要": "模拟病例：术后规律随访，最近一次记录日期为 2024-07-15。",
        },
        "时序随访基础信息": [
            "随访时间: 2024-02-15, 2024-04-15, 2024-07-15",
            "术后天数: 31, 91, 182", "年龄: 1.3, 1.5, 1.7", "身高: 74, 77, 81", "体重: 9.8, 10.6, 11.4",
        ],
        "时序检验指标 (纯数值序列)": [f"{name}: {values}" for name, values in LAB_VALUES.items()],
        "时序用药记录 (纯数值序列)": medications,
        "临床事件": ["模拟超声随访：肝内血流可见。", "模拟门诊记录：继续按医嘱复诊。"],
        "真实标签": {"1m": None, "1y": None, "5y": None},
    }


def non_missing_rows(static: np.ndarray, temporal: np.ndarray, mask: np.ndarray, schema: dict) -> list[dict]:
    rows = []
    for index, value in enumerate(static):
        if np.isfinite(value):
            rows.append({"block": "static", "time_step": "", "feature_index": index, "feature_name": schema["static_feature_names"][index], "value": float(value)})
    for step in np.where(mask)[0]:
        for feature_index, value in enumerate(temporal[step]):
            if np.isfinite(value):
                rows.append({"block": "temporal", "time_step": int(step), "feature_index": feature_index, "feature_name": schema["temporal_feature_names"][feature_index], "value": float(value)})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block", "time_step", "feature_index", "feature_name", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    patient = synthetic_patient()
    (RESEARCH_ROOT / "llm" / "examples" / "patient_raw_example.json").write_text(
        json.dumps(patient, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prompt = build_final_prompt(patient)
    (RESEARCH_ROOT / "llm" / "examples" / "patient_llm_input_example.txt").write_text(prompt + "\n", encoding="utf-8")
    (RESEARCH_ROOT / "llm" / "examples" / "inference_prompt_example.json").write_text(
        json.dumps(
            {
                "system": SYSTEM_PROMPT,
                "user": prompt,
                "assistant_expected_output": {"1m": 0.12, "1y": 0.26, "5y": 0.41},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    schema_path = RESEARCH_ROOT / "machine_learning" / "examples" / "schema_public.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    static = np.asarray(encode_static(patient, schema), dtype=np.float32)
    temporal, mask, raw_length, truncated = parse_temporal(patient, schema, int(schema["max_seq_len"]))
    flat = np.concatenate([static, temporal.reshape(-1), mask.astype(np.float32), np.asarray([raw_length], dtype=np.float32)])
    rows = non_missing_rows(static, temporal, mask, schema)
    examples_dir = RESEARCH_ROOT / "machine_learning" / "examples"
    write_csv(examples_dir / "xgboost_input_example.csv", rows)
    write_csv(examples_dir / "rsf_input_example.csv", rows)
    write_csv(examples_dir / "lstm_input_example.csv", rows)
    np.save(examples_dir / "xgboost_input_vector.npy", flat)
    np.save(examples_dir / "rsf_input_vector.npy", flat)
    np.savez(
        examples_dir / "lstm_input_tensors.npz",
        static=static,
        temporal=temporal,
        time_mask=mask,
        sequence_lengths=np.asarray([raw_length], dtype=np.int32),
    )
    (examples_dir / "input_shapes.json").write_text(
        json.dumps(
            {
                "synthetic_only": True,
                "static_shape": [int(static.shape[0])],
                "temporal_shape": [int(value) for value in temporal.shape],
                "time_mask_shape": [int(mask.shape[0])],
                "raw_sequence_length": int(raw_length),
                "truncated": bool(truncated),
                "xgboost_and_rsf_flat_shape": [int(flat.shape[0])],
                "flattening_order": "static + temporal.reshape(-1) + time_mask + sequence_length",
                "identifier_used_by_preprocessing_but_not_model_input": patient["id"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
