import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from sequence_common import (
    CONTINUOUS_BASE_COLS,
    DATASET_DIR,
    HORIZON_KEYS,
    KNOWN_CATEGORICAL_BASE_COLS,
    REPO_ROOT,
    TARGETS,
    TEST_JSON,
    TRAIN_JSON,
    clean_num,
    label_to_float,
    load_json,
    patient_id,
    split_series_field,
    survival_lookup_from_items,
    visit_count,
    write_json,
)


def build_schema(train_items):
    cat_cols = set(KNOWN_CATEGORICAL_BASE_COLS)
    categorical_values = defaultdict(set)
    temporal_base_cols = set()
    lab_cols = set()
    med_cols = set()

    for item in train_items:
        base = item.get("基础信息", {})
        for key in base:
            if str(key).startswith("诊断"):
                cat_cols.add(key)

        for col in cat_cols:
            value = str(base.get(col, "")).strip()
            if value not in {"", "NaN", "None", "nan"}:
                categorical_values[col].add(value)

        for field in item.get("时序随访基础信息", []):
            name, _ = split_series_field(field)
            if name and name != "随访时间":
                temporal_base_cols.add(name)

        for field in item.get("时序检验指标 (纯数值序列)", []):
            name, _ = split_series_field(field)
            if name:
                lab_cols.add(name)

        for field in item.get("时序用药记录 (纯数值序列)", []):
            name, _ = split_series_field(field)
            if name:
                med_cols.add(name)

    static_feature_names = []
    static_feature_names.extend([f"Base_{col}" for col in CONTINUOUS_BASE_COLS])
    static_feature_names.extend(["Base_患儿性别_男1女0", "Base_供体性别_男1女0"])
    categorical_values_out = {
        col: sorted(values) for col, values in sorted(categorical_values.items())
    }
    for col, values in categorical_values_out.items():
        static_feature_names.extend([f"Cat_{col}_{value}" for value in values])

    temporal_specs = []
    for name in sorted(temporal_base_cols):
        temporal_specs.append({"type": "temporal_base", "name": name, "feature_name": f"BaseSeq_{name}"})
    for name in sorted(lab_cols):
        temporal_specs.append({"type": "lab", "name": name, "feature_name": f"LabSeq_{name}"})
    for name in sorted(med_cols):
        temporal_specs.append({"type": "med", "name": name, "feature_name": f"MedSeq_{name}"})

    return {
        "repo_root": str(REPO_ROOT),
        "source_train_json": str(TRAIN_JSON),
        "source_test_json": str(TEST_JSON),
        "continuous_base_cols": list(CONTINUOUS_BASE_COLS),
        "categorical_base_values": categorical_values_out,
        "static_feature_names": static_feature_names,
        "temporal_feature_specs": temporal_specs,
        "temporal_feature_names": [spec["feature_name"] for spec in temporal_specs],
        "target_names": TARGETS,
        "horizon_keys": HORIZON_KEYS,
        "representation": "raw longitudinal values by timestep; no Latest/Mean/Max/Min/Count aggregation",
    }


def encode_static(item, schema):
    base = item.get("基础信息", {})
    row = []
    for col in schema["continuous_base_cols"]:
        try:
            row.append(float(base.get(col)))
        except Exception:
            row.append(np.nan)

    gender = str(base.get("患儿性别")).strip()
    row.append(1.0 if gender == "男" else (0.0 if gender == "女" else np.nan))
    donor_gender = str(base.get("供体性别")).strip()
    row.append(1.0 if donor_gender == "男" else (0.0 if donor_gender == "女" else np.nan))

    for col, values in schema["categorical_base_values"].items():
        value = str(base.get(col)).strip()
        valid = value not in {"", "NaN", "None", "nan"}
        for candidate in values:
            row.append(1.0 if value == candidate else (0.0 if valid else np.nan))
    return row


def raw_sequence_length(item):
    lengths = []
    for section in ["时序随访基础信息", "时序检验指标 (纯数值序列)", "时序用药记录 (纯数值序列)"]:
        for field in item.get(section, []):
            _, values = split_series_field(field)
            if values:
                lengths.append(len(values))
    return max(lengths) if lengths else 1


def parse_temporal(item, schema, max_seq_len):
    specs = schema["temporal_feature_specs"]
    feature_lookup = {(spec["type"], spec["name"]): idx for idx, spec in enumerate(specs)}
    raw_len = raw_sequence_length(item)
    matrix = np.full((raw_len, len(specs)), np.nan, dtype=np.float32)

    def fill_section(section_name, spec_type):
        for field in item.get(section_name, []):
            name, values = split_series_field(field)
            idx = feature_lookup.get((spec_type, name))
            if idx is None:
                continue
            for pos, value in enumerate(values[:raw_len]):
                if spec_type == "med":
                    matrix[pos, idx] = 0.0 if value in {"", "NaN", "None", "nan"} else 1.0
                elif value not in {"", "NaN", "None", "nan"}:
                    matrix[pos, idx] = clean_num(value)

    fill_section("时序随访基础信息", "temporal_base")
    fill_section("时序检验指标 (纯数值序列)", "lab")
    fill_section("时序用药记录 (纯数值序列)", "med")

    truncated = raw_len > max_seq_len
    if truncated:
        matrix = matrix[-max_seq_len:, :]
        effective_len = max_seq_len
    else:
        effective_len = raw_len

    padded = np.full((max_seq_len, len(specs)), np.nan, dtype=np.float32)
    mask = np.zeros(max_seq_len, dtype=bool)
    padded[-effective_len:, :] = matrix
    mask[-effective_len:] = True
    return padded, mask, raw_len, truncated


def label_counts(labels):
    out = {}
    for idx, target in enumerate(TARGETS):
        col = labels[:, idx]
        valid = ~np.isnan(col)
        out[target] = {
            "negative": int(np.sum(col[valid] == 0)),
            "positive": int(np.sum(col[valid] == 1)),
            "missing": int(np.sum(~valid)),
        }
    return out


def sequence_stats(lengths):
    lengths = np.asarray(lengths, dtype=float)
    return {
        "min": int(np.min(lengths)),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(np.max(lengths)),
    }


def transform_items(items, schema, max_seq_len, split_name):
    n = len(items)
    static_dim = len(schema["static_feature_names"])
    temporal_dim = len(schema["temporal_feature_names"])

    static = np.full((n, static_dim), np.nan, dtype=np.float32)
    temporal = np.full((n, max_seq_len, temporal_dim), np.nan, dtype=np.float32)
    time_mask = np.zeros((n, max_seq_len), dtype=bool)
    labels = np.full((n, len(TARGETS)), np.nan, dtype=np.float32)
    sample_ids = []
    patient_ids = []
    visit_counts = np.zeros(n, dtype=np.int32)
    sequence_lengths = np.zeros(n, dtype=np.int32)
    truncated = np.zeros(n, dtype=bool)

    for idx, item in enumerate(items):
        if idx and idx % 10000 == 0:
            print(f"  {split_name}: parsed {idx}/{n}")
        sid = str(item.get("id", ""))
        labels_dict = item.get("真实标签", {})
        sample_ids.append(sid)
        patient_ids.append(patient_id(sid))
        visit_counts[idx] = visit_count(sid)
        labels[idx] = [label_to_float(labels_dict.get(key)) for key in HORIZON_KEYS]
        static[idx] = np.asarray(encode_static(item, schema), dtype=np.float32)
        temporal[idx], time_mask[idx], sequence_lengths[idx], truncated[idx] = parse_temporal(
            item, schema, max_seq_len
        )

    return {
        "static": static,
        "temporal": temporal,
        "time_mask": time_mask,
        "labels": labels,
        "sample_ids": np.asarray(sample_ids, dtype="U80"),
        "patient_ids": np.asarray(patient_ids, dtype="U80"),
        "visit_counts": visit_counts,
        "sequence_lengths": sequence_lengths,
        "truncated": truncated,
    }


def save_dataset(path, data, compressed=False):
    saver = np.savez_compressed if compressed else np.savez
    saver(path, **data)


def save_survival_targets(path, items):
    lookup = survival_lookup_from_items(items)
    sample_ids = []
    events = []
    times = []
    for item in items:
        sid = str(item.get("id", ""))
        event, time = lookup[sid]
        sample_ids.append(sid)
        events.append(event)
        times.append(time)
    np.savez(
        path,
        sample_ids=np.asarray(sample_ids, dtype="U80"),
        event=np.asarray(events, dtype=bool),
        time=np.asarray(times, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading train JSON: {TRAIN_JSON}")
    train_items = load_json(TRAIN_JSON)
    print(f"Loading test JSON: {TEST_JSON}")
    test_items = load_json(TEST_JSON)

    train_patients = {patient_id(item.get("id", "")) for item in train_items}
    test_patients = {patient_id(item.get("id", "")) for item in test_items}
    overlap = train_patients & test_patients
    if overlap:
        raise RuntimeError(f"Patient overlap detected between train/test: {len(overlap)}")

    print("Building train-only schema")
    schema = build_schema(train_items)
    schema["max_seq_len"] = args.max_seq_len
    write_json(DATASET_DIR / "schema.json", schema)

    print("Transforming train")
    train = transform_items(train_items, schema, args.max_seq_len, "train")
    print("Transforming test")
    test = transform_items(test_items, schema, args.max_seq_len, "test")

    save_dataset(DATASET_DIR / "sequence_train.npz", train, compressed=args.compressed)
    save_dataset(DATASET_DIR / "sequence_test.npz", test, compressed=args.compressed)
    save_survival_targets(DATASET_DIR / "survival_train.npz", train_items)
    save_survival_targets(DATASET_DIR / "survival_test.npz", test_items)

    report = {
        "max_seq_len": args.max_seq_len,
        "compressed_npz": bool(args.compressed),
        "train_rows": int(len(train_items)),
        "test_rows": int(len(test_items)),
        "train_patients": int(len(train_patients)),
        "test_patients": int(len(test_patients)),
        "patient_overlap": int(len(overlap)),
        "static_dim": int(train["static"].shape[1]),
        "temporal_dim_per_step": int(train["temporal"].shape[2]),
        "flattened_temporal_dim": int(train["temporal"].shape[1] * train["temporal"].shape[2]),
        "train_sequence_length": sequence_stats(train["sequence_lengths"]),
        "test_sequence_length": sequence_stats(test["sequence_lengths"]),
        "train_truncated_n": int(np.sum(train["truncated"])),
        "test_truncated_n": int(np.sum(test["truncated"])),
        "train_label_counts": label_counts(train["labels"]),
        "test_label_counts": label_counts(test["labels"]),
        "notes": [
            "Temporal input keeps raw per-visit values by position.",
            "No Latest/Mean/Max/Min/Count aggregation features are generated.",
            "Padding positions are NaN in temporal arrays and False in time_mask.",
        ],
    }
    write_json(DATASET_DIR / "dataset_report.json", report)
    print(f"Saved sequence datasets under {DATASET_DIR}")
    print(f"static_dim={report['static_dim']}, temporal_dim={report['temporal_dim_per_step']}")
    print(f"train_truncated_n={report['train_truncated_n']}, test_truncated_n={report['test_truncated_n']}")


if __name__ == "__main__":
    main()
