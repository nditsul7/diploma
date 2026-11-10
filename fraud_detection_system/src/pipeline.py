"""Об'єднаний пайплайн передбачення: ознаки → Isolation Forest → XGBoost → рішення."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.feature_engineering import RAW_COLUMNS
from src.train import ANOMALY_COL, BASELINE_FILES

ARTIFACTS = ("feature_engineer.joblib", "iso_forest.joblib", "xgboost_model.joblib", "config.json")


class FraudPipeline:
    def __init__(self, fe, iso_forest, xgb_model, config: dict, baselines: dict | None = None) -> None:
        self.fe = fe
        self.iso_forest = iso_forest
        self.xgb_model = xgb_model
        self.config = config
        self.thresholds = config["thresholds"]
        self.feature_columns = config["feature_columns"]
        self.baselines = baselines or {}  # Random Forest, LightGBM, XGBoost без anomaly_score
        self.model_thresholds = config.get("model_thresholds", {})
        # Для онлайн-запитів (1 рядок) паралелізм лише додає накладні витрати
        self.iso_forest.n_jobs = 1
        self.xgb_model.set_params(n_jobs=1)
        for model in self.baselines.values():
            model.set_params(n_jobs=1)

    @classmethod
    def load(cls, models_dir: str | Path = "models") -> "FraudPipeline":
        d = Path(models_dir)
        missing = [a for a in ARTIFACTS if not (d / a).exists()]
        if missing:
            raise FileNotFoundError(
                f"У {d} немає артефактів {missing}. Спершу навчіть моделі: python -m src.train"
            )
        baselines = {k: joblib.load(d / f) for k, f in BASELINE_FILES.items() if (d / f).exists()}
        return cls(
            joblib.load(d / "feature_engineer.joblib"),
            joblib.load(d / "iso_forest.joblib"),
            joblib.load(d / "xgboost_model.joblib"),
            json.loads((d / "config.json").read_text(encoding="utf-8")),
            baselines,
        )

    def _features(self, raw: pd.DataFrame) -> pd.DataFrame:
        X = self.fe.transform(raw)
        X[ANOMALY_COL] = self.iso_forest.score_samples(X)
        return X[self.feature_columns]  # той самий порядок, що й під час навчання

    def _decide(self, prob: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        t = self.thresholds
        return np.select(
            [prob >= t["block"], (prob >= t["review"]) | (anomaly <= t["anomaly_review"])],
            ["BLOCK", "REVIEW"],
            default="ALLOW",
        )

    def predict(self, raw: pd.DataFrame, explain_top_k: int = 0) -> tuple[pd.DataFrame, list | None]:
        """Повертає таблицю рішень і (опційно) пояснення для кожного рядка."""
        X = self._features(raw[RAW_COLUMNS])
        prob = self.xgb_model.predict_proba(X)[:, 1]
        anomaly = X[ANOMALY_COL].to_numpy()
        result = pd.DataFrame(
            {
                "fraud_probability": np.round(prob, 4),
                "anomaly_score": np.round(anomaly, 4),
                "is_fraud": prob >= self.thresholds["review"],
                "action": self._decide(prob, anomaly),
            },
            index=raw.index,
        )
        return result, (self._explain(X, explain_top_k) if explain_top_k else None)

    def predict_all(self, raw: pd.DataFrame) -> dict[str, np.ndarray]:
        """Скори всіх моделей порівняння (для оцінки якості на власних пакетах)."""
        X = self._features(raw[RAW_COLUMNS])
        plain = [c for c in self.feature_columns if c != ANOMALY_COL]
        out = {key: model.predict_proba(X[plain])[:, 1] for key, model in self.baselines.items()}
        out["hybrid"] = self.xgb_model.predict_proba(X)[:, 1]
        out["isolation_forest_only"] = -X[ANOMALY_COL].to_numpy()  # менше score = аномальніше
        return out

    def _explain(self, X: pd.DataFrame, top_k: int) -> list[list[dict]]:
        """Вклад ознак у log-odds (значення SHAP для дерев, нативно в XGBoost)."""
        contribs = self.xgb_model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        out = []
        for row_values, row_contrib in zip(X.to_numpy(), contribs):
            top = np.argsort(-np.abs(row_contrib))[:top_k]
            out.append(
                [
                    {
                        "feature": X.columns[i],
                        "value": round(float(row_values[i]), 4),
                        "contribution": round(float(row_contrib[i]), 4),
                    }
                    for i in top
                ]
            )
        return out