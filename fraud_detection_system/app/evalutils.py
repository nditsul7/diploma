"""Метрики класифікації на numpy (без sklearn): ROC/PR-криві, матриця помилок, F1 за порогом."""
from __future__ import annotations

import numpy as np


def roc_pr_curves(y: np.ndarray, score: np.ndarray):
    """ROC і PR-криві. Повертає fpr, tpr, roc_auc, precision, recall, average_precision."""
    order = np.argsort(-score, kind="mergesort")
    y_s, s_s = y[order], score[order]
    idx = np.r_[np.where(np.diff(s_s))[0], len(y_s) - 1]
    tps = np.cumsum(y_s)[idx]
    fps = 1 + idx - tps
    pos, neg = tps[-1], fps[-1]
    tpr, fpr = np.r_[0.0, tps / pos], np.r_[0.0, fps / neg]
    prec, rec = tps / (tps + fps), tps / pos
    roc_auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return fpr, tpr, roc_auc, np.r_[1.0, prec], np.r_[0.0, rec], ap


def confusion(y: np.ndarray, flagged: np.ndarray) -> dict:
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def f1_sweep(y: np.ndarray, p: np.ndarray) -> dict:
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    rows = [confusion(y, p >= t) for t in grid]
    return {
        "thresholds": grid.tolist(),
        "precision": [r["precision"] for r in rows],
        "recall": [r["recall"] for r in rows],
        "f1": [r["f1"] for r in rows],
    }


def thin_pair(a: np.ndarray, b: np.ndarray, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Проріджує криву до n точок (для швидкого малювання)."""
    if len(a) <= n:
        return a, b
    idx = np.unique(np.linspace(0, len(a) - 1, n).astype(int))
    return a[idx], b[idx]