from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class MLResult(BaseModel):
    status: str = Field(description="ok | error | unavailable")
    death_probability_1m: Optional[float] = None
    death_probability_1y: Optional[float] = None
    death_probability_5y: Optional[float] = None
    error: Optional[str] = None


class LLMResult(BaseModel):
    status: str = Field(description="ok | partial | error | disabled")
    death_probability_1m: Optional[float] = None
    death_probability_1y: Optional[float] = None
    death_probability_5y: Optional[float] = None
    rationale: Optional[str] = None
    rationale_source: Optional[str] = None
    patterns: list[dict[str, str]] = Field(default_factory=list)
    pattern_output: Optional[str] = None
    pair_count: Optional[int] = None
    parse_status: Optional[str] = None
    parse_warnings: list[str] = Field(default_factory=list)
    basic_info_pattern_filter: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class PredictionResult(BaseModel):
    id: Optional[str] = None
    ml: MLResult
    llm: LLMResult


class BatchPredictionResponse(BaseModel):
    count: int
    results: list[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    backend: str
    ml: dict[str, Any]
    llm: dict[str, Any]


class JobSummary(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    filename: Optional[str] = None
    total: int = 0
    completed: int = 0
    failed: int = 0
    error: Optional[str] = None


class JobResultsResponse(BaseModel):
    job_id: str
    status: str
    total: int
    results: list[PredictionResult]
