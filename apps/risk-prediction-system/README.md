# PediLT-Risk Prediction System

[中文说明](README.zh-CN.md)

Browser-based research system for pediatric liver transplantation follow-up mortality-risk prediction. It accepts standard longitudinal records and reports independent XGBoost and LLM estimates for 1 month, 1 year, and 5 years.

> Research demonstration only. Not for diagnosis, treatment, prognosis communication, or clinical decision-making.

## Architecture

```text
React/nginx -> FastAPI -> input adapter
                         ├─ bundled XGBoost models
                         └─ optional OpenAI-compatible vLLM API
```

The ML and LLM branches are never averaged or fused. If the LLM returns `null` for a horizon, the API, page, JSON download, and CSV download preserve `null`; no ML value is used as a replacement.

## Bundled ML artifacts

The backend image includes the validated schema and three XGBoost models under `artifacts/`. See [`artifacts/model_manifest.json`](artifacts/model_manifest.json) for checksums and the 22,927-feature runtime contract.

The models were trained on private clinical records that are not distributed. Only aggregate validation information is published.

## Docker deployment

### ML-only or external vLLM

```bash
cp .env.example .env
docker compose up -d --build
```

- Web UI: <http://localhost:8080>
- Backend API: <http://localhost:8000>
- API documentation: <http://localhost:8000/docs>

For an external OpenAI-compatible service, configure:

```dotenv
LLM_BASE_URL=https://your-vllm-host.example/v1
LLM_API_KEY=EMPTY
LLM_MODEL_NAME=zeno156/PediLT-Risk-Qwen3-4B
LLM_PROMPT_MODE=sft_pattern_prob
```

Leaving `LLM_BASE_URL` blank runs the application in ML-only mode.

### Optional local vLLM profile

Requirements: Linux Docker host, NVIDIA driver, NVIDIA Container Toolkit, and sufficient GPU memory for the selected context length.

```bash
# Set HF_TOKEN only when the Hugging Face repository requires authentication.
export HF_TOKEN="replace-with-hugging-face-token"
docker compose -f compose.yaml -f compose.vllm.yaml --profile llm up -d --build
```

The local OpenAI-compatible vLLM endpoint is exposed at <http://localhost:8001/v1>. Adjust `VLLM_MAX_MODEL_LEN` and `VLLM_TENSOR_PARALLEL_SIZE` for the available hardware.

## Input formats

- `.json`: one patient object, an array, or `{ "patients": [...] }`
- `.jsonl`: one patient object per line
- `.csv` / `.xlsx`: one patient per row; a `patient_json` column may contain the complete JSON record

Two explicitly synthetic JSON examples are available under `frontend/public/samples/`. They contain no real identity, outcome label, or exact clinical date.

## Main API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Backend, ML, and LLM status |
| `POST` | `/api/predict` | One-patient ML + LLM prediction |
| `POST` | `/api/batch_predict` | Batch prediction from JSON |
| `POST` | `/api/jobs` | Asynchronous file upload job |
| `GET` | `/api/jobs/{job_id}` | Job status |
| `GET` | `/api/jobs/{job_id}/results` | Paginated results |
| `GET` | `/api/jobs/{job_id}/download` | CSV or JSON download |

Uploaded files and results are stored only in the local Docker volume. Do not expose this research service to untrusted networks without authentication, TLS, access control, retention controls, and an institutional privacy review.

## Local development and tests

```powershell
./scripts/run_backend.ps1
./scripts/run_frontend.ps1
```

```bash
cd backend
python -m pytest

cd ../frontend
npm ci
npm test
npm run build
```
