"""Генератор синтетичного датасету зі схемою Kaggle Credit Card Fraud.

Дає змогу запустити весь проєкт без завантаження реальних даних.
Щоб працювати з реальним датасетом, просто покладіть його у
data/raw_transactions.csv (колонки Time, V1..V28, Amount, Class).

Запуск:
    python -m src.generate_data --rows 100000 --fraud-rate 0.006
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import V_COLUMNS

# Компоненти, у яких шахрайські транзакції зсунуті відносно легітимних
FRAUD_SHIFTS = {
    "V1": -3.0, "V2": 2.5, "V3": -4.0, "V4": 3.0, "V7": -2.0, "V9": -1.5,
    "V10": -3.5, "V11": 2.5, "V12": -4.0, "V14": -5.0, "V16": -2.0, "V17": -3.0,
}


def generate(n_rows: int = 100_000, fraud_rate: float = 0.006, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_rows * fraud_rate), 10)
    n_legit = n_rows - n_fraud

    # Дисперсія компонент спадає, як у PCA-ознаках реального датасету
    scales = np.linspace(2.0, 0.5, len(V_COLUMNS))

    # --- Легітимні транзакції ---
    legit_v = rng.normal(0.0, scales, size=(n_legit, len(V_COLUMNS)))
    legit_amount = rng.lognormal(mean=3.5, sigma=1.3, size=n_legit)
    # Добовий ритм: більшість операцій удень
    legit_hour = np.clip(rng.normal(14, 4.5, size=n_legit), 0, 23.99)
    legit_time = rng.integers(0, 2, size=n_legit) * 86400 + legit_hour * 3600

    # --- Шахрайські транзакції ---
    shifts = np.array([FRAUD_SHIFTS.get(c, 0.0) for c in V_COLUMNS])
    # 30% «складних» випадків із ослабленим зсувом, щоб задача не була тривіальною
    hard = rng.random(n_fraud) < 0.3
    strength = np.where(hard, 0.35, 1.0)[:, None]
    fraud_v = rng.normal(0.0, scales * 1.3, size=(n_fraud, len(V_COLUMNS))) + shifts * strength
    # Суми: дрібні «тестові» списання або великі
    small = rng.random(n_fraud) < 0.5
    fraud_amount = np.where(
        small,
        rng.lognormal(1.5, 0.8, size=n_fraud),
        rng.lognormal(5.5, 1.0, size=n_fraud),
    )
    # Шахраї частіше діють вночі
    night = rng.random(n_fraud) < 0.55
    fraud_hour = np.where(night, rng.uniform(23, 29, size=n_fraud) % 24, rng.uniform(6, 22, size=n_fraud))
    fraud_time = rng.integers(0, 2, size=n_fraud) * 86400 + fraud_hour * 3600

    df = pd.DataFrame(np.vstack([legit_v, fraud_v]), columns=V_COLUMNS)
    df.insert(0, "Time", np.concatenate([legit_time, fraud_time]))
    df["Amount"] = np.round(np.concatenate([legit_amount, fraud_amount]), 2)
    df["Class"] = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    return df.sort_values("Time").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--fraud-rate", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw_transactions.csv")
    args = parser.parse_args()

    df = generate(args.rows, args.fraud_rate, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Збережено {len(df)} рядків ({int(df['Class'].sum())} шахрайських) у {out}")


if __name__ == "__main__":
    main()
