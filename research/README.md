# PediLT-Risk research package

[中文说明](README.zh-CN.md)

Reproducibility-oriented code for node-level mortality-risk prediction after pediatric liver transplantation. The experiments compare structured machine-learning models and large language models at 1-month, 1-year, and 5-year horizons.

This directory contains code, configuration templates, aggregate results, and fully synthetic examples only. It does not contain EHR records, patient-level predictions, raw LLM logs, SFT datasets, or training checkpoints.

## Contents

- `machine_learning/train`: preprocessing, XGBoost, LSTM, RSF, and evaluation scripts.
- `llm/sft`: SFT dataset construction, patient-level splitting, and Qwen3-4B LoRA launcher.
- `llm/grpo`: GRPO training and probability reward functions.
- `llm/inference`: OpenAI-compatible inference client.
- `prompts`: versioned SFT, GRPO, and inference contracts.
- `evaluation`: aggregate metrics and patient-cluster bootstrap analyses.
- `machine_learning/examples` and `llm/examples`: explicitly synthetic public fixtures.

## Data and feature contract

Each sample represents one follow-up node and uses information available up to that node. Patient-level splits prevent nodes from the same patient crossing train and test sets.

- Static features: 142
- Temporal features per step: 88
- Maximum sequence length: 256
- Flattened XGBoost/RSF representation: 22,927
- Targets: `Label_1m`, `Label_1y`, and `Label_5y`

Set `PEDILT_DATA_DIR` and the other variables documented in `configs/paths.env.example` to authorized locations outside the repository. Never place clinical data inside this repository.

## Main workflows

```bash
python machine_learning/train/preprocessing/build_sequence_datasets.py --max-seq-len 256
python machine_learning/train/xgboost/train_xgboost_sequence.py
python machine_learning/train/lstm/train_lstm_sequence.py
python machine_learning/train/rsf/train_rsf_sequence.py
python machine_learning/train/evaluation/summarize_results.py
```

See `machine_learning/train/TRAINING_PROTOCOL.md` for the sequence representation, input contract, and expected outputs.

For the LLM pipeline, build patient-disjoint SFT data, launch Qwen3-4B LoRA training, and run inference through an approved OpenAI-compatible endpoint. See `docs/llm_pipeline.md` and `llm/sft/README.md`.

The released fine-tuned model is available at [zeno156/PediLT-Risk-Qwen3-4B](https://huggingface.co/zeno156/PediLT-Risk-Qwen3-4B). Probability `null` is a valid model output and must not be imputed from ML predictions.

## Aggregate evaluation

Authoritative aggregate tables are under `evaluation/metrics/`. Common-cohort sample sizes are 88,640 nodes for 1 month, 76,684 for 1 year, and 35,079 for 5 years. Patient-level prediction files required to recompute significance tests are private and are not distributed.

## Public smoke test

```bash
python -m unittest discover -s tests -v
python -m compileall llm machine_learning evaluation tests
```

Full training and metric reproduction require separately authorized access to the original clinical data and the private prediction files described in the documentation.
