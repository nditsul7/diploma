"""REST API на FastAPI для оцінки транзакцій у реальному часі."""
from __future__ import annotations

import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.feature_engineering import V_COLUMNS
from src.pipeline import FraudPipeline

logger = logging.getLogger("fraud_api")


class _TransactionBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # одруківка на кшталт "v1" дасть 422, а не мовчазний нуль
        json_schema_extra={"example": {"Time": 3600.0, "Amount": 1250.5, "V1": -1.35, "V2": 0.42}},
    )
    Time: float = Field(..., ge=0, allow_inf_nan=False, description="Секунди від початку спостереження")
    Amount: float = Field(..., ge=0, allow_inf_nan=False, description="Сума транзакції")


# V1..V28: якщо поле не передано, використовується 0.0 (нейтральне значення PCA-компоненти)
TransactionRequest = create_model(
    "TransactionRequest",
    __base__=_TransactionBase,
    **{c: (float, Field(0.0, allow_inf_nan=False)) for c in V_COLUMNS},
)


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float = Field(..., description="Вклад у log-odds шахрайства (+ підвищує ризик)")


class FraudPredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    anomaly_score: float = Field(..., description="Менше значення = аномальніша транзакція")
    action: Literal["ALLOW", "REVIEW", "BLOCK"]
    explanation: list[FeatureContribution] | None = None


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    results: list[FraudPredictionResponse]


class CompareResponse(BaseModel):
    probabilities: dict[str, list[float]] = Field(..., description="Імовірність шахрайства від кожної моделі")
    thresholds: dict[str, float] = Field(..., description="Пороги з навчання (максимум F1 на валідації)")


def create_app(models_dir: str | None = None) -> FastAPI:
    models_path = Path(models_dir or os.getenv("MODELS_DIR", "models"))
    api_key = os.getenv("API_KEY", "")  # порожній = авторизація API вимкнена

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and not hmac.compare_digest(x_api_key or "", api_key):
            raise HTTPException(status_code=401, detail="Невірний або відсутній X-API-Key")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = FraudPipeline.load(models_path)
        logger.info("Моделі завантажено")
        yield

    app = FastAPI(
        title="Intelligent Fraud Detection API",
        description="Інтелектуальна система виявлення аномалій та шахрайства у транзакціях",
        version="1.2.0",
        lifespan=lifespan,
    )

    def _run(request: Request, rows: list[dict], top_k: int) -> list[dict]:
        pipeline: FraudPipeline = request.app.state.pipeline
        try:
            result, explanations = pipeline.predict(pd.DataFrame(rows), explain_top_k=top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        records = result.to_dict("records")
        if explanations is not None:
            for rec, expl in zip(records, explanations):
                rec["explanation"] = expl
        return records

    @app.post("/predict", response_model=FraudPredictionResponse, dependencies=[Depends(require_key)])
    def predict(
        tx: TransactionRequest,  # type: ignore[valid-type]
        request: Request,
        explain: bool = Query(False, description="Додати топ ознак, що вплинули на рішення"),
        top_k: int = Query(5, ge=1, le=20),
    ):
        return _run(request, [tx.model_dump()], top_k if explain else 0)[0]

    @app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(require_key)])
    def predict_batch(batch: BatchRequest, request: Request):
        rows = [t.model_dump() for t in batch.transactions]
        return {"results": _run(request, rows, 0)}

    @app.post("/predict/compare", response_model=CompareResponse, dependencies=[Depends(require_key)])
    def predict_compare(batch: BatchRequest, request: Request):
        """Прогнози всіх моделей порівняння для пакета (оцінка якості на власних даних)."""
        pipeline: FraudPipeline = request.app.state.pipeline
        try:
            probs = pipeline.predict_all(pd.DataFrame([t.model_dump() for t in batch.transactions]))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "probabilities": {k: np.round(v, 5).tolist() for k, v in probs.items()},
            "thresholds": pipeline.model_thresholds,
        }

    @app.get("/health")
    def health(request: Request):
        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Моделі не завантажено")
        return {
            "status": "ok",
            "service": "fraud-detection-api",
            "trained_at": pipeline.config.get("trained_at"),
            "thresholds": pipeline.thresholds,
        }

    @app.get("/metrics", dependencies=[Depends(require_key)])
    def metrics():
        """Метрики якості з останнього навчання (ROC/PR-криві, F1, матриці помилок)."""
        path = models_path / "metrics.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="metrics.json не знайдено, спершу навчіть моделі")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return response

    return app


app = create_app()