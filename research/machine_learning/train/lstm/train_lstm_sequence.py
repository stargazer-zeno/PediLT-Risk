import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

PREPROCESS_DIR = Path(__file__).resolve().parents[1] / "preprocessing"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from sequence_common import (
    TARGETS,
    TRAIN_ROOT,
    format_metric,
    load_schema,
    load_sequence_npz,
    score_binary,
    set_seed,
    write_json,
)


OUT_DIR = TRAIN_ROOT / "lstm"
PRED_CSV = OUT_DIR / "lstm_sequence_saved_test_predictions.csv"
METRICS_JSON = OUT_DIR / "lstm_sequence_metrics.json"
REPORT_TXT = OUT_DIR / "lstm_sequence_report.txt"
PREPROCESS_NPZ = OUT_DIR / "lstm_sequence_preprocess.npz"


class SequenceDataset(Dataset):
    def __init__(self, static, temporal, lengths, labels, sample_ids, patient_ids, visit_counts):
        self.static = torch.from_numpy(static.astype(np.float32, copy=False))
        self.temporal = torch.from_numpy(temporal.astype(np.float32, copy=False))
        self.lengths = torch.from_numpy(lengths.astype(np.int64, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.float32, copy=False))
        self.sample_ids = sample_ids
        self.patient_ids = patient_ids
        self.visit_counts = visit_counts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.static[idx], self.temporal[idx], self.lengths[idx], self.labels[idx], idx


class PackedHybridLSTM(nn.Module):
    def __init__(self, static_dim, temporal_dim, lstm_hidden=96, static_hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(temporal_dim, lstm_hidden, num_layers=1, batch_first=True)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, static_hidden),
            nn.BatchNorm1d(static_hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden + static_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, static_x, temporal_x, lengths):
        static_out = self.static_mlp(static_x)
        packed = pack_padded_sequence(
            temporal_x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        lstm_out = hidden[-1]
        return self.classifier(torch.cat([static_out, lstm_out], dim=1)).squeeze(1)


def temporal_continuous_indices(schema):
    return np.asarray(
        [idx for idx, spec in enumerate(schema["temporal_feature_specs"]) if spec["type"] != "med"],
        dtype=np.int64,
    )


def left_to_right_padded(temporal, lengths):
    out = np.zeros_like(temporal, dtype=np.float32)
    for idx, length in enumerate(lengths):
        length = int(length)
        out[idx, :length, :] = temporal[idx, -length:, :]
    return out


def fit_preprocess(train, schema):
    static = train["static"].astype(np.float32).copy()
    static_means = np.nanmean(static, axis=0)
    static_means = np.nan_to_num(static_means, nan=0.0).astype(np.float32)
    missing = np.isnan(static)
    static[missing] = np.take(static_means, np.where(missing)[1])
    static_scaler = StandardScaler().fit(static)

    temporal = train["temporal"].astype(np.float32)
    cont_idx = temporal_continuous_indices(schema)
    cont = temporal[:, :, cont_idx].reshape(-1, len(cont_idx))
    temporal_means = np.nanmean(cont, axis=0)
    temporal_means = np.nan_to_num(temporal_means, nan=0.0).astype(np.float32)
    temporal_stds = np.nanstd(cont, axis=0)
    temporal_stds = np.where(temporal_stds > 1e-6, temporal_stds, 1.0).astype(np.float32)
    return static_means, static_scaler.mean_.astype(np.float32), static_scaler.scale_.astype(np.float32), cont_idx, temporal_means, temporal_stds


def apply_preprocess(data, static_means, static_center, static_scale, cont_idx, temporal_means, temporal_stds):
    static = data["static"].astype(np.float32).copy()
    missing = np.isnan(static)
    static[missing] = np.take(static_means, np.where(missing)[1])
    static = np.nan_to_num(static, nan=0.0)
    static = ((static - static_center) / static_scale).astype(np.float32)

    temporal = data["temporal"].astype(np.float32).copy()
    if len(cont_idx):
        cont = temporal[:, :, cont_idx]
        nan_mask = np.isnan(cont)
        if nan_mask.any():
            cont[nan_mask] = np.take(temporal_means, np.where(nan_mask)[2])
        cont -= temporal_means.reshape(1, 1, -1)
        cont /= temporal_stds.reshape(1, 1, -1)
        temporal[:, :, cont_idx] = cont

    temporal = np.nan_to_num(temporal, nan=0.0).astype(np.float32)
    temporal = left_to_right_padded(temporal, data["sequence_lengths"])
    return static, temporal


def make_train_valid_split(labels, patient_ids, valid_size, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    train_idx, valid_idx = next(splitter.split(np.zeros(len(labels)), labels, groups=patient_ids))
    if len(np.unique(labels[valid_idx])) >= 2:
        return train_idx, valid_idx

    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    for _ in range(100):
        valid_patients = set(
            rng.choice(unique_patients, size=max(1, int(len(unique_patients) * valid_size)), replace=False)
        )
        valid_mask = np.asarray([pid in valid_patients for pid in patient_ids])
        valid_idx = np.where(valid_mask)[0]
        train_idx = np.where(~valid_mask)[0]
        if len(valid_idx) and len(np.unique(labels[valid_idx])) == 2:
            return train_idx, valid_idx
    return train_idx, valid_idx


def predict(model, loader, device):
    model.eval()
    probs, labels, row_indices = [], [], []
    with torch.no_grad():
        for static, temporal, lengths, y, idx in loader:
            logits = model(static.to(device), temporal.to(device), lengths.to(device))
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
            row_indices.extend(idx.numpy().tolist())
    return np.asarray(labels), np.asarray(probs), np.asarray(row_indices, dtype=int)


def safe_auroc(labels, probs):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return score_binary(labels, probs)["auroc"]


def train_one(target, target_idx, train, test, train_static, train_temporal, test_static, test_temporal, args, device):
    y_train_raw = train["labels"][:, target_idx]
    y_test_raw = test["labels"][:, target_idx]
    train_mask = ~np.isnan(y_train_raw)
    test_mask = ~np.isnan(y_test_raw)
    y_train = y_train_raw[train_mask].astype(np.float32)
    y_test = y_test_raw[test_mask].astype(np.float32)

    patient_ids = train["patient_ids"].astype(str)[train_mask]
    train_idx, valid_idx = make_train_valid_split(y_train, patient_ids, args.valid_size, args.seed + target_idx)

    train_ds = SequenceDataset(
        train_static[train_mask][train_idx],
        train_temporal[train_mask][train_idx],
        train["sequence_lengths"][train_mask][train_idx],
        y_train[train_idx],
        train["sample_ids"].astype(str)[train_mask][train_idx],
        patient_ids[train_idx],
        train["visit_counts"][train_mask][train_idx],
    )
    valid_ds = SequenceDataset(
        train_static[train_mask][valid_idx],
        train_temporal[train_mask][valid_idx],
        train["sequence_lengths"][train_mask][valid_idx],
        y_train[valid_idx],
        train["sample_ids"].astype(str)[train_mask][valid_idx],
        patient_ids[valid_idx],
        train["visit_counts"][train_mask][valid_idx],
    )
    test_ds = SequenceDataset(
        test_static[test_mask],
        test_temporal[test_mask],
        test["sequence_lengths"][test_mask],
        y_test,
        test["sample_ids"].astype(str)[test_mask],
        test["patient_ids"].astype(str)[test_mask],
        test["visit_counts"][test_mask],
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed + target_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = PackedHybridLSTM(
        static_dim=train_static.shape[1],
        temporal_dim=train_temporal.shape[2],
        lstm_hidden=args.lstm_hidden,
        static_hidden=args.static_hidden,
    ).to(device)

    pos = float(np.sum(y_train[train_idx] == 1))
    neg = float(np.sum(y_train[train_idx] == 0))
    pos_weight = torch.tensor([neg / pos if pos else 1.0], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    losses = []
    valid_aurocs = []
    best_state = None
    best_epoch = 0
    best_valid_auroc = -np.inf
    wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for static, temporal, lengths, labels, _ in train_loader:
            optimizer.zero_grad()
            logits = model(static.to(device), temporal.to(device), lengths.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item())

        losses.append(total_loss / max(len(train_loader), 1))
        valid_labels, valid_probs, _ = predict(model, valid_loader, device)
        valid_auroc = safe_auroc(valid_labels, valid_probs)
        valid_aurocs.append(valid_auroc)
        if np.isfinite(valid_auroc) and valid_auroc > best_valid_auroc + args.min_delta:
            best_valid_auroc = valid_auroc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
        print(f"{target} epoch={epoch} loss={losses[-1]:.5f} valid_auroc={format_metric(valid_auroc)}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = OUT_DIR / f"lstm_sequence_{target}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target": target,
            "config": vars(args),
            "static_dim": int(train_static.shape[1]),
            "temporal_dim": int(train_temporal.shape[2]),
        },
        model_path,
    )

    labels, probs, row_indices = predict(model, test_loader, device)
    pred_df = pd.DataFrame(
        {
            "Target": target,
            "Sample_ID": test_ds.sample_ids[row_indices],
            "Patient_ID": test_ds.patient_ids[row_indices],
            "Visit_Count": test_ds.visit_counts[row_indices],
            "True_Label": labels,
            "Pred_Prob": probs,
        }
    )
    metrics = score_binary(labels, probs)
    metrics.update(
        {
            "loss_by_epoch": losses,
            "valid_auroc_by_epoch": valid_aurocs,
            "best_epoch": int(best_epoch),
            "best_valid_auroc": float(best_valid_auroc) if np.isfinite(best_valid_auroc) else None,
            "train_n": int(len(train_ds)),
            "valid_n": int(len(valid_ds)),
            "model_path": str(model_path),
        }
    )
    return metrics, pred_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lstm-hidden", type=int, default=96)
    parser.add_argument("--static-hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    schema = load_schema()
    train = load_sequence_npz("train")
    test = load_sequence_npz("test")

    print("Fitting preprocessors on train only")
    static_means, static_center, static_scale, cont_idx, temporal_means, temporal_stds = fit_preprocess(train, schema)
    np.savez(
        PREPROCESS_NPZ,
        static_means=static_means,
        static_center=static_center,
        static_scale=static_scale,
        temporal_continuous_indices=cont_idx,
        temporal_means=temporal_means,
        temporal_stds=temporal_stds,
    )

    print("Applying preprocessors")
    train_static, train_temporal = apply_preprocess(
        train, static_means, static_center, static_scale, cont_idx, temporal_means, temporal_stds
    )
    test_static, test_temporal = apply_preprocess(
        test, static_means, static_center, static_scale, cont_idx, temporal_means, temporal_stds
    )

    metrics = {
        "config": {
            **vars(args),
            "device": str(device),
            "model": "PackedHybridLSTM",
            "input_representation": "raw timestep matrix with sequence lengths; packed LSTM ignores padding",
            "max_seq_len": schema["max_seq_len"],
            "static_dim": int(train_static.shape[1]),
            "temporal_dim_per_step": int(train_temporal.shape[2]),
            "torch_version": torch.__version__,
            "preprocess_path": str(PREPROCESS_NPZ),
        },
        "test": {},
    }
    report_lines = [f"device={device}", f"max_seq_len={schema['max_seq_len']}"]
    pred_parts = []

    for target_idx, target in enumerate(TARGETS):
        print(f"Training {target}")
        target_metrics, pred_df = train_one(
            target,
            target_idx,
            train,
            test,
            train_static,
            train_temporal,
            test_static,
            test_temporal,
            args,
            device,
        )
        metrics["test"][target] = target_metrics
        pred_parts.append(pred_df)
        report_lines.append(
            f"{target}: AUROC={format_metric(target_metrics['auroc'])}, "
            f"n={target_metrics['n']}, "
            f"pos={target_metrics['positive_n']}, best_epoch={target_metrics['best_epoch']}, "
            f"valid_AUROC={format_metric(target_metrics['best_valid_auroc'])}"
        )

    pd.concat(pred_parts, ignore_index=True).to_csv(PRED_CSV, index=False)
    write_json(METRICS_JSON, metrics)
    REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
