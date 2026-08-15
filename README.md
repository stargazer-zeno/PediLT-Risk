# PediLT-Risk

[中文说明](README.zh-CN.md)

PediLT-Risk is an open-source research project for dynamic mortality-risk prediction after pediatric liver transplantation. It contains a deployable prediction system, model artifacts, and research code for the machine-learning and large-language-model experiments.

> **Research use only.** This software is not a medical device and must not be used for diagnosis, treatment, or individual clinical decisions. See [DISCLAIMER.md](DISCLAIMER.md).

## Repository layout

- [`apps/risk-prediction-system`](apps/risk-prediction-system): FastAPI + React application with bundled XGBoost models and optional vLLM integration.
- [`research`](research): training, inference, evaluation, prompts, aggregate metrics, and synthetic examples.

The models estimate mortality risk at three horizons: 1 month, 1 year, and 5 years. The ML and LLM branches are reported independently; they are not averaged or fused. A missing LLM probability remains `null` and is never replaced by an ML prediction.

## Quick start

```bash
cd apps/risk-prediction-system
cp .env.example .env
docker compose up -d --build
```

Open <http://localhost:8080>. The default configuration starts the web application and bundled XGBoost service. To connect an external OpenAI-compatible vLLM endpoint, set `LLM_BASE_URL` and `LLM_MODEL_NAME` in `.env`.

For an optional local NVIDIA GPU vLLM service:

```bash
docker compose -f compose.yaml -f compose.vllm.yaml --profile llm up -d --build
```

LLM weights: [zeno156/PediLT-Risk-Qwen3-4B](https://huggingface.co/zeno156/PediLT-Risk-Qwen3-4B).

## Data availability

The source clinical records are private and are not distributed. This repository contains aggregate evaluation results and explicitly marked synthetic examples only. Reproducing the reported cohort metrics requires separately authorized access to the original data.

## Validation

```bash
cd apps/risk-prediction-system/backend
python -m pytest

cd ../frontend
npm ci
npm test
npm run build

cd ../../../research
python -m unittest discover -s tests -v
```

## License and citation

Code is released under the [Apache License 2.0](LICENSE). Model artifacts and third-party components remain subject to their documented licenses. Citation metadata is provided in [CITATION.cff](CITATION.cff).
