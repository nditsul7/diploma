"""Бібліотека масивів транзакційних даних користувача: CSV на диску + index.json."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(os.getenv("DATASETS_DIR", "data/datasets"))
V_COLS = [f"V{i}" for i in range(1, 29)]
INPUT_COLS = ["Time", "Amount", *V_COLS]


def _user_dir(username: str) -> Path:
    d = ROOT / re.sub(r"[^A-Za-z0-9_.-]", "_", username)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_index(d: Path) -> dict:
    f = d / "index.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _save_index(d: Path, idx: dict) -> None:
    (d / "index.json").write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_name(name: str) -> str:
    stem = re.sub(r"\.csv$", "", name.strip(), flags=re.I)
    stem = re.sub(r"[^\w.\- ]", "_", stem, flags=re.U).strip(" .")[:60]
    if not stem:
        raise ValueError("Вкажіть назву масиву")
    return stem


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Перевіряє масив: Time, Amount обов'язкові; V1–V28 за відсутності = 0."""
    missing = [c for c in ("Time", "Amount") if c not in df.columns]
    if missing:
        raise ValueError(f"Відсутні колонки: {missing}")
    df = df.copy()
    for col in V_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[INPUT_COLS] = df[INPUT_COLS].apply(pd.to_numeric, errors="coerce")
    if len(df) == 0:
        raise ValueError("Масив порожній")
    if df[INPUT_COLS].isna().any().any():
        raise ValueError("У даних є пропущені або нечислові значення в колонках Time/Amount/V1–V28")
    if (df["Time"] < 0).any() or (df["Amount"] < 0).any():
        raise ValueError("Time та Amount не можуть бути від'ємними")
    return df.reset_index(drop=True)


def list_datasets(username: str) -> list[dict]:
    d = _user_dir(username)
    items = [{"name": n, **m} for n, m in _load_index(d).items() if (d / f"{n}.csv").exists()]
    return sorted(items, key=lambda m: m["saved_at"], reverse=True)


def find_by_sha(username: str, sha: str) -> str | None:
    return next((m["name"] for m in list_datasets(username) if m.get("sha") == sha), None)


def save_dataset(username: str, name: str, df: pd.DataFrame, source: str = "csv", sha: str | None = None) -> str:
    name, df = clean_name(name), normalize(df)
    d = _user_dir(username)
    df.to_csv(d / f"{name}.csv", index=False)
    frauds = None
    if "Class" in df.columns:
        labels = pd.to_numeric(df["Class"], errors="coerce")
        frauds = int((labels == 1).sum())
    idx = _load_index(d)
    idx[name] = {
        "rows": len(df), "frauds": frauds, "source": source, "sha": sha,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_index(d, idx)
    return name


def load_dataset(username: str, name: str) -> pd.DataFrame:
    return pd.read_csv(_user_dir(username) / f"{clean_name(name)}.csv")


def delete_dataset(username: str, name: str) -> None:
    d = _user_dir(username)
    name = clean_name(name)
    (d / f"{name}.csv").unlink(missing_ok=True)
    idx = _load_index(d)
    idx.pop(name, None)
    _save_index(d, idx)