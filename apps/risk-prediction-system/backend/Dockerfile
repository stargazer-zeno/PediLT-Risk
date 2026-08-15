FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for xgboost (libgomp) etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY artifacts /artifacts

# The validated schema and XGBoost models are bundled in the image.
ENV SCHEMA_PATH=/artifacts/datasets/schema.json \
    MODEL_DIR=/artifacts/xgboost \
    JOBS_DIR=/data/jobs

EXPOSE 8000

# 1 worker keeps the in-process JobStore consistent. Scale out via an external
# queue (Redis/Celery) if you need multiple workers.
CMD ["gunicorn", "app.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "300"]
