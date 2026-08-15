from __future__ import annotations

import csv
import io
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..aggregator import flatten_for_csv

LOGGER = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

CSV_COLUMNS = [
    "id",
    "ml_status",
    "ml_death_probability_1m",
    "ml_death_probability_1y",
    "ml_death_probability_5y",
    "ml_error",
    "llm_status",
    "llm_death_probability_1m",
    "llm_death_probability_1y",
    "llm_death_probability_5y",
    "llm_rationale",
    "llm_rationale_source",
    "llm_pattern_output",
    "llm_patterns",
    "llm_pair_count",
    "llm_parse_status",
    "llm_parse_warnings",
    "llm_error",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """File-backed async job store.

    Each job lives under ``<jobs_dir>/<job_id>/`` with ``meta.json``,
    ``input.json``, ``results.json`` and ``results.csv``. A thread pool runs the
    predictions; this is sufficient for a single-process deployment. For
    multi-instance scaling, swap this for Redis/Celery + PostgreSQL.
    """

    def __init__(self, jobs_dir: str | Path, max_workers: int = 2):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def _meta_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "meta.json"

    def _write_meta(self, meta: dict[str, Any]) -> None:
        path = self._meta_path(meta["job_id"])
        with self._lock:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

    def _read_meta(self, job_id: str) -> dict[str, Any] | None:
        path = self._meta_path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def create_job(
        self,
        patients: list[dict[str, Any]],
        predict_batch: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        filename: str | None = None,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input.json").write_text(
            json.dumps(patients, ensure_ascii=False), encoding="utf-8"
        )
        meta = {
            "job_id": job_id,
            "status": STATUS_PENDING,
            "created_at": _now(),
            "updated_at": _now(),
            "filename": filename,
            "total": len(patients),
            "completed": 0,
            "failed": 0,
            "error": None,
        }
        self._write_meta(meta)
        self._executor.submit(self._run_job, job_id, patients, predict_batch)
        return meta

    def _run_job(
        self,
        job_id: str,
        patients: list[dict[str, Any]],
        predict_batch: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> None:
        meta = self._read_meta(job_id) or {}
        meta.update(status=STATUS_RUNNING, updated_at=_now())
        self._write_meta(meta)
        try:
            results = predict_batch(patients)
            self._save_results(job_id, results)
            failed = sum(
                1
                for r in results
                if (r.get("ml", {}).get("status") not in {"ok"})
                and (r.get("llm", {}).get("status") not in {"ok", "disabled"})
            )
            meta.update(
                status=STATUS_COMPLETED,
                updated_at=_now(),
                completed=len(results),
                failed=failed,
            )
            self._write_meta(meta)
            LOGGER.info("Job %s completed (%d records).", job_id, len(results))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Job %s failed", job_id)
            meta.update(status=STATUS_FAILED, updated_at=_now(), error=str(exc))
            self._write_meta(meta)

    def _save_results(self, job_id: str, results: list[dict[str, Any]]) -> None:
        job_dir = self._job_dir(job_id)
        (job_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (job_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for result in results:
                writer.writerow(flatten_for_csv(result))

    # ---------------------------------------------------------------- queries
    def get_status(self, job_id: str) -> dict[str, Any] | None:
        return self._read_meta(job_id)

    def get_results(self, job_id: str) -> list[dict[str, Any]] | None:
        path = self._job_dir(job_id) / "results.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def get_csv_path(self, job_id: str) -> Path | None:
        path = self._job_dir(job_id) / "results.csv"
        return path if path.exists() else None

    def get_json_path(self, job_id: str) -> Path | None:
        path = self._job_dir(job_id) / "results.json"
        return path if path.exists() else None

    def results_to_csv_string(self, results: list[dict[str, Any]]) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(flatten_for_csv(result))
        return buffer.getvalue()

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for child in self.jobs_dir.iterdir():
            if child.is_dir():
                meta = self._read_meta(child.name)
                if meta:
                    jobs.append(meta)
        jobs.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return jobs

    def shutdown(self) -> None:  # pragma: no cover
        self._executor.shutdown(wait=False)
