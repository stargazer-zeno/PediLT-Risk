from __future__ import annotations

import logging
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .adapters.standard_adapter import StandardDataAdapter, StandardDataAdapterError
from .aggregator import PredictionAggregator
from .config import get_settings
from .jobs.store import JobStore
from .llm.llm_service import LLMService
from .ml.ml_service import MLService
from .schemas import BatchPredictionResponse, HealthResponse, PredictionResult

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
LOGGER = logging.getLogger("pedilt_risk.gateway")

settings = get_settings()

app = FastAPI(
    title="PediLT-Risk Prediction API",
    version="1.0.0",
    description=(
        "Browser-facing gateway that standardizes follow-up data and returns "
        "XGBoost (ML) and remote LLM mortality risk for 1 month / 1 year / 5 years."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_service = MLService(settings.ml)
llm_service = LLMService(settings.llm)
aggregator = PredictionAggregator(ml_service, llm_service)
adapter = StandardDataAdapter()
job_store = JobStore(settings.jobs_dir, max_workers=settings.job_workers)


# ---------------------------------------------------- compatibility XGBoost API
@app.get("/health")
def compatibility_health() -> dict[str, str]:
    """Compatible with the original XGBoost-only service health check."""
    if not ml_service.available:
        raise HTTPException(status_code=503, detail="XGBoost model unavailable.")
    return {"status": "ok"}


@app.post("/predict")
def compatibility_predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Compatible single-patient XGBoost prediction (ML only)."""
    result = ml_service.predict_one(payload)
    if result["status"] == "unavailable":
        raise HTTPException(status_code=503, detail=result.get("error"))
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {k: v for k, v in result.items() if k.startswith("death_probability")}


@app.post("/batch_predict")
def compatibility_batch_predict(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatible batch XGBoost prediction (ML only)."""
    results = ml_service.predict_batch(payload)
    if results and results[0]["status"] == "unavailable":
        raise HTTPException(status_code=503, detail=results[0].get("error"))
    return [
        {k: v for k, v in r.items() if k.startswith("death_probability")}
        for r in results
    ]


# ------------------------------------------------------------------- new API
@app.get("/api/health", response_model=HealthResponse)
def api_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": "ok",
        "ml": ml_service.health(),
        "llm": llm_service.health(),
    }


@app.post("/api/predict", response_model=PredictionResult)
def api_predict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        patients = adapter.from_json_obj(payload)
    except StandardDataAdapterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if len(patients) != 1:
        raise HTTPException(
            status_code=400,
            detail="/api/predict expects exactly one patient. Use /api/batch_predict.",
        )
    return aggregator.predict_one(patients[0])


@app.post("/api/batch_predict", response_model=BatchPredictionResponse)
def api_batch_predict(payload: Any = Body(...)) -> dict[str, Any]:
    try:
        patients = adapter.from_json_obj(payload)
    except StandardDataAdapterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    results = aggregator.predict_batch(patients)
    return {"count": len(results), "results": results}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds limit of {settings.max_upload_mb} MB.",
        )
    try:
        patients = adapter.from_bytes(content, file.filename or "upload.json")
    except StandardDataAdapterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    meta = job_store.create_job(
        patients, aggregator.predict_batch, filename=file.filename
    )
    return meta


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": job_store.list_jobs()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    meta = job_store.get_status(job_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return meta


@app.get("/api/jobs/{job_id}/results")
def get_job_results(job_id: str) -> dict[str, Any]:
    meta = job_store.get_status(job_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    results = job_store.get_results(job_id)
    if results is None:
        raise HTTPException(
            status_code=409,
            detail=f"Results not ready. Current status: {meta.get('status')}.",
        )
    return {
        "job_id": job_id,
        "status": meta.get("status"),
        "total": meta.get("total", len(results)),
        "results": results,
    }


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str, format: str = Query("csv", pattern="^(csv|json)$")):
    meta = job_store.get_status(job_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if format == "csv":
        path = job_store.get_csv_path(job_id)
        media_type = "text/csv"
    else:
        path = job_store.get_json_path(job_id)
        media_type = "application/json"
    if path is None:
        raise HTTPException(
            status_code=409,
            detail=f"Results not ready. Current status: {meta.get('status')}.",
        )
    return FileResponse(
        path, media_type=media_type, filename=f"{job_id}_results.{format}"
    )


@app.exception_handler(StandardDataAdapterError)
def _adapter_error_handler(_request, exc: StandardDataAdapterError):  # pragma: no cover
    return JSONResponse(status_code=400, content={"detail": str(exc)})
