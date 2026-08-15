# Longitudinal sequence model training protocol

## Scope

The machine-learning pipeline evaluates three model families on the same
patient-disjoint longitudinal cohort:

- LSTM consumes variable-length timestep matrices with sequence lengths and
  padding masks.
- XGBoost and Random Survival Forest consume fixed-width vectors formed by
  flattening the raw timestep matrix and appending the mask and sequence
  length.

The tree-based representations do not add hand-crafted longitudinal summary
features such as `Latest`, `Mean`, `Max`, `Min`, or `Count`.

## Input contract

Set `PEDILT_DATA_DIR` to an authorized data directory outside the repository.
The preprocessing command expects:

- `train_dataset_gold.json`
- `test_dataset_gold.json`

Samples must be split by patient before preprocessing. Temporal values are
right-aligned to `MAX_SEQ_LEN=256`; padding values are stored as `NaN`, and
padding positions are `False` in `time_mask`.

## Commands

Run the following commands from the `research` directory in an environment
that provides the dependencies listed in `requirements.txt`:

```bash
python machine_learning/train/preprocessing/build_sequence_datasets.py --max-seq-len 256
python machine_learning/train/xgboost/train_xgboost_sequence.py
python machine_learning/train/lstm/train_lstm_sequence.py
python machine_learning/train/rsf/train_rsf_sequence.py
python machine_learning/train/evaluation/summarize_results.py
```

## Outputs

Runtime outputs are written below `machine_learning/train/` and are ignored by
Git. They include preprocessed NPZ datasets, fitted model artifacts,
patient-level predictions, metrics, and summary tables. Do not commit outputs
derived from clinical records.
