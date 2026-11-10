"""Обробка даних та формування ознак для транзакцій.

Схема вхідних даних збігається з датасетом Kaggle «Credit Card Fraud Detection»:
Time, Amount, V1..V28 (та Class для навчання).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time", "Amount", *V_COLUMNS]
ENGINEERED_COLUMNS = ["hour_of_day", "is_night", "log_amount", "scaled_amount"]
# Єдиний порядок колонок для всіх моделей (навчання і прогноз).
FEATURE_COLUMNS = [*RAW_COLUMNS, *ENGINEERED_COLUMNS]

SECONDS_IN_DAY = 24 * 3600


class TransactionFeatureEngineer:
    """Очищення, часові ознаки та стійке до викидів масштабування суми.

    Масштабувальник навчається лише методом ``fit`` (на навчальній вибірці),
    тому тестові та «бойові» дані не впливають на його параметри.
    """

    def __init__(self) -> None:
        self.scaler = RobustScaler()

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Відсутні обов'язкові колонки: {missing}")
        if df[RAW_COLUMNS].isna().any().any():
            raise ValueError("Вхідні дані містять пропущені значення (NaN)")
        if (df["Amount"] < 0).any():
            raise ValueError("Amount не може бути від'ємним")

    def fit(self, df: pd.DataFrame) -> "TransactionFeatureEngineer":
        self._validate(df)
        self.scaler.fit(df[["Amount"]].astype("float64"))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate(df)
        out = df[RAW_COLUMNS].astype("float64").copy()

        # Часові ознаки: година доби та нічний прапорець
        out["hour_of_day"] = (out["Time"] % SECONDS_IN_DAY) / 3600.0
        out["is_night"] = ((out["hour_of_day"] >= 23) | (out["hour_of_day"] <= 5)).astype("float64")

        # Сума: логарифм і робастне масштабування (стійкість до викидів)
        out["log_amount"] = np.log1p(out["Amount"])
        out["scaled_amount"] = self.scaler.transform(out[["Amount"]]).ravel()

        return out[FEATURE_COLUMNS]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
