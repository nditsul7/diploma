"""Налаштування користувача: власні пороги рішень і параметри сповіщень (JSON-файл на користувача)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.getenv("SETTINGS_DIR", "data/user_settings"))
DEFAULTS = {
    "thresholds": {"custom": False, "review": None, "block": None, "block_enabled": True},
    "notify": {"on_block": True, "email_enabled": False, "email": "", "push_enabled": False, "push_topic": ""},
}


def _path(username: str) -> Path:
    return ROOT / (re.sub(r"[^A-Za-z0-9_.-]", "_", username) + ".json")


def load(username: str) -> dict:
    try:
        data = json.loads(_path(username).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {k: {**default, **(data.get(k) or {})} for k, default in DEFAULTS.items()}


def save(username: str, cfg: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _path(username).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def differs(server: dict, review: float, block: float) -> bool:
    """Чи відрізняються пороги від серверних (якщо сервер їх не повернув — вважаємо, що відрізняються)."""
    return not (abs(server.get("review", -1) - review) < 1e-9 and abs(server.get("block", -1) - block) < 1e-9)


def effective_thresholds(server: dict, cfg: dict) -> dict:
    """Пороги, якими користується дашборд: власні (якщо збережені й відрізняються) поверх порогів з API."""
    t = cfg["thresholds"]
    th = dict(server)
    th["custom"] = False
    if t["custom"] and t["review"] is not None and t["block"] is not None:
        th["review"], th["block"] = float(t["review"]), float(t["block"])
        th["custom"] = differs(server, th["review"], th["block"])
    th["block_enabled"] = bool(t["block_enabled"])
    return th


def classify(p, review: float, block: float, block_enabled: bool = True) -> np.ndarray:
    """Рішення за ймовірністю: BLOCK / REVIEW / ALLOW. Без автоблокування BLOCK стає REVIEW."""
    p = np.asarray(p, dtype=float)
    action = np.where(p >= block, "BLOCK", np.where(p >= review, "REVIEW", "ALLOW"))
    if not block_enabled:
        action = np.where(action == "BLOCK", "REVIEW", action)
    return action


def decide(server_action: str, prob: float, th: dict) -> str:
    """Рішення для однієї транзакції з урахуванням власних порогів."""
    if th.get("custom") and "review" in th and "block" in th:
        action = str(classify([prob], th["review"], th["block"])[0])
    else:
        action = server_action
    if action == "BLOCK" and not th.get("block_enabled", True):
        action = "REVIEW"
    return action


def reclassify(df: pd.DataFrame, th: dict) -> pd.DataFrame:
    """Перераховує колонку action для масиву за власними порогами (модель повторно не викликається)."""
    out = df.copy()
    if th.get("custom") and "review" in th and "block" in th:
        out["action"] = classify(out["fraud_probability"].to_numpy(), th["review"], th["block"])
    if not th.get("block_enabled", True):
        out["action"] = out["action"].where(out["action"] != "BLOCK", "REVIEW")
    return out