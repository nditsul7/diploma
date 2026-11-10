"""Навчання гібридної моделі: Isolation Forest (1-й рівень) + XGBoost (2-й рівень).

Запуск з кореня проєкту:
    python -m src.train --data data/raw_transactions.csv --models-dir models
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier

try:  # LightGBM потрібен лише для порівняльної таблиці
    from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
except ImportError:
    LGBMClassifier = None

from src.feature_engineering import RAW_COLUMNS, TransactionFeatureEngineer

ANOMALY_COL = "anomaly_score"

BASELINE_FILES = {
    "random_forest": "random_forest.joblib",
    "lightgbm": "lightgbm.joblib",
    "xgboost_without_anomaly_score": "xgboost_plain.joblib",
}
MODEL_NAMES = {
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost_without_anomaly_score": "XGBoost (без Isolation Forest)",
    "hybrid": "Гібридна модель (Isolation Forest + XGBoost)",
}

# Вартість помилок для підбору порогів (умовні одиниці, налаштовуються)
COST_FN = 100.0  # пропущене шахрайство
COST_FP = 5.0    # зайве втручання (перевірка / тертя для клієнта)
BLOCK_PRECISION = 0.95  # мінімальна точність для автоматичного BLOCK
ANOMALY_REVIEW_QUANTILE = 0.005  # 0.5% найаномальніших → REVIEW
CURVE_POINTS = 400  # скільки точок кривої зберігати в metrics.json


def _new_iso_forest(seed: int, n_estimators: int = 100) -> IsolationForest:
    return IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)


def oof_isolation_scores(X: pd.DataFrame, seed: int, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold anomaly score для навчальної вибірки.

    Кожну транзакцію оцінює Isolation Forest, який її НЕ бачив. Інакше на
    навчанні score був би оптимістичнішим (точки «знайомі» лісу), ніж на
    нових даних, і XGBoost навчився б на зміщеній ознаці.
    """
    scores = np.zeros(len(X))
    for fit_idx, score_idx in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        model = _new_iso_forest(seed).fit(X.iloc[fit_idx])
        scores[score_idx] = model.score_samples(X.iloc[score_idx])
    return scores


def pick_thresholds(y_val: np.ndarray, p_val: np.ndarray) -> dict[str, float]:
    """Пороги підбираються на валідації (без SMOTE) за вартістю помилок."""
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    pos = y_val == 1

    costs = [COST_FN * np.sum(pos & (p_val < t)) + COST_FP * np.sum(~pos & (p_val >= t)) for t in grid]
    review = float(grid[int(np.argmin(costs))])

    block = None
    for t in grid[grid >= review]:
        flagged = p_val >= t
        if flagged.sum() > 0 and np.sum(flagged & pos) / flagged.sum() >= BLOCK_PRECISION:
            block = float(t)
            break
    if block is None:
        block = max(review, 0.9)
    return {"review": review, "block": block}


def _confusion(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0.0,
    }


def _thin(*arrays: np.ndarray, n: int = CURVE_POINTS) -> list[list[float]]:
    """Рівномірно проріджує криву, щоб metrics.json не розбухав."""
    length = len(arrays[0])
    idx = np.arange(length) if length <= n else np.unique(np.linspace(0, length - 1, n).astype(int))
    return [np.round(np.asarray(a, dtype=float)[idx], 5).tolist() for a in arrays]


def _curves(y: np.ndarray, score: np.ndarray) -> dict:
    """ROC- і Precision-Recall криві для довільного «скору шахрайства»."""
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)
    f, t = _thin(fpr, tpr)
    p, r = _thin(precision, recall)
    return {
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "pr_auc": round(float(average_precision_score(y, score)), 4),
        "roc": {"fpr": f, "tpr": t},
        "pr": {"precision": p, "recall": r},
    }


def f1_by_threshold(y: np.ndarray, p: np.ndarray) -> dict[str, list[float]]:
    """Precision / Recall / F1 залежно від порога ймовірності."""
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    prec, rec, f1 = [], [], []
    for t in grid:
        c = _confusion(y, p >= t)
        prec.append(c["precision"])
        rec.append(c["recall"])
        f1.append(c["f1"])
    return {"thresholds": grid.tolist(), "precision": prec, "recall": rec, "f1": f1}


def best_f1_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Поріг, що максимізує F1 на валідації (однаково для всіх моделей порівняння)."""
    sweep = f1_by_threshold(y_val, p_val)
    return float(sweep["thresholds"][int(np.argmax(sweep["f1"]))])


def comparison_row(key: str, y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray) -> dict:
    """Рядок порівняльної таблиці: поріг з валідації, метрики на тесті."""
    thr = best_f1_threshold(y_val, p_val)
    c = _confusion(y_test, p_test >= thr)
    return {
        "key": key,
        "name": MODEL_NAMES[key],
        "threshold": thr,
        "precision": c["precision"],
        "recall": c["recall"],
        "f1": c["f1"],
        "roc_auc": round(float(roc_auc_score(y_test, p_test)), 4),
        "pr_auc": round(float(average_precision_score(y_test, p_test)), 4),
    }


def _make_xgb(seed: int, n_estimators: int, early_stopping: bool) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="aucpr",
        early_stopping_rounds=30 if early_stopping else None,
        n_jobs=-1,
    )


def train_pipeline(
    data_path: str,
    models_dir: str = "models",
    seed: int = 42,
    xgb_n_estimators: int = 400,
    use_smote: bool = True,
) -> dict:
    df = pd.read_csv(data_path)
    missing = [c for c in [*RAW_COLUMNS, "Class"] if c not in df.columns]
    if missing:
        raise ValueError(f"У датасеті немає колонок: {missing}")
    X_raw, y = df[RAW_COLUMNS], df["Class"].astype(int)

    # 1. Спершу розділення, потім fit ознак: тест не впливає на масштабувальник
    X_rest, X_test_raw, y_rest, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.2, random_state=seed, stratify=y_rest
    )

    fe = TransactionFeatureEngineer()
    X_train = fe.fit_transform(X_train_raw)
    X_val = fe.transform(X_val_raw)
    X_test = fe.transform(X_test_raw)

    # 2. Рівень 1: Isolation Forest (без міток)
    train_scores = oof_isolation_scores(X_train, seed)  # out-of-fold
    iso = _new_iso_forest(seed).fit(X_train)            # фінальна модель для нових даних
    val_scores = iso.score_samples(X_val)
    test_scores = iso.score_samples(X_test)
    anomaly_review_threshold = float(np.quantile(train_scores, ANOMALY_REVIEW_QUANTILE))

    X_train = X_train.assign(**{ANOMALY_COL: train_scores})
    X_val = X_val.assign(**{ANOMALY_COL: val_scores})
    X_test = X_test.assign(**{ANOMALY_COL: test_scores})

    # 3. SMOTE лише на навчальній вибірці
    if use_smote:
        X_fit, y_fit = SMOTE(sampling_strategy=0.1, random_state=seed).fit_resample(X_train, y_train)
    else:
        X_fit, y_fit = X_train, y_train

    # 4. Рівень 2: XGBoost, ранню зупинку рахуємо на справжньому (незбалансованому) валідаційному наборі
    xgb = _make_xgb(seed, xgb_n_estimators, early_stopping=True)
    xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    # 5. Пороги: на валідації; оцінка якості: на тесті
    p_val = xgb.predict_proba(X_val)[:, 1]
    thresholds = pick_thresholds(y_val.to_numpy(), p_val)
    p_test = xgb.predict_proba(X_test)[:, 1]
    y_t = y_test.to_numpy()

    # Абляція: чи справді anomaly_score щось додає?
    xgb_plain = _make_xgb(seed, xgb_n_estimators, early_stopping=True)
    cols = [c for c in X_fit.columns if c != ANOMALY_COL]
    xgb_plain.fit(X_fit[cols], y_fit, eval_set=[(X_val[cols], y_val)], verbose=False)
    p_plain = xgb_plain.predict_proba(X_test[cols])[:, 1]

    block_flag = p_test >= thresholds["block"]
    review_flag = (p_test >= thresholds["review"]) | (test_scores <= anomaly_review_threshold)

    # Криві для дашборда (ROC, Precision-Recall, F1 залежно від порога)
    # Базові моделі для порівняння: ті самі дані та SMOTE, але без anomaly_score
    p_plain_val = xgb_plain.predict_proba(X_val[cols])[:, 1]
    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
    )
    rf.fit(X_fit[cols], y_fit)
    model_probs = {"random_forest": (rf.predict_proba(X_val[cols])[:, 1], rf.predict_proba(X_test[cols])[:, 1])}
    fitted = {"random_forest": rf, "xgboost_without_anomaly_score": xgb_plain}
    if LGBMClassifier is not None:
        lgbm = LGBMClassifier(
            n_estimators=xgb_n_estimators, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1,
        )
        lgbm.fit(
            X_fit[cols], y_fit, eval_set=[(X_val[cols], y_val)], eval_metric="average_precision",
            callbacks=[lgb_early_stopping(30, verbose=False)],
        )
        model_probs["lightgbm"] = (lgbm.predict_proba(X_val[cols])[:, 1], lgbm.predict_proba(X_test[cols])[:, 1])
        fitted["lightgbm"] = lgbm
    else:
        print("УВАГА: lightgbm не встановлено (pip install lightgbm), LightGBM пропущено у порівнянні")
    model_probs["xgboost_without_anomaly_score"] = (p_plain_val, p_plain)
    model_probs["hybrid"] = (p_val, p_test)

    y_v = y_val.to_numpy()
    comparison = [comparison_row(k, y_v, pv, y_t, pt) for k, (pv, pt) in model_probs.items()]
    curves = {
        k: {**_curves(y_t, pt), "f1_by_threshold": f1_by_threshold(y_t, pt)} for k, (_, pt) in model_probs.items()
    }
    curves["isolation_forest_only"] = _curves(y_t, -test_scores)  # менше score = аномальніше
    importance = sorted(zip(X_fit.columns, xgb.feature_importances_), key=lambda kv: -kv[1])[:15]

    metrics = {
        "rows": len(df),
        "fraud_rate": round(float(y.mean()), 5),
        "test_size": len(y_test),
        "test_frauds": int(y_t.sum()),
        "hybrid": {
            "roc_auc": curves["hybrid"]["roc_auc"],
            "pr_auc": curves["hybrid"]["pr_auc"],
        },
        "xgboost_without_anomaly_score": {
            "roc_auc": curves["xgboost_without_anomaly_score"]["roc_auc"],
            "pr_auc": curves["xgboost_without_anomaly_score"]["pr_auc"],
        },
        "isolation_forest_only": {
            "roc_auc": curves["isolation_forest_only"]["roc_auc"],
            "pr_auc": curves["isolation_forest_only"]["pr_auc"],
        },
        "at_review_threshold": _confusion(y_t, review_flag),
        "at_block_threshold": _confusion(y_t, block_flag),
        "action_share": {
            "BLOCK": round(float(block_flag.mean()), 5),
            "REVIEW": round(float((review_flag & ~block_flag).mean()), 5),
            "ALLOW": round(float((~review_flag & ~block_flag).mean()), 5),
        },
        "feature_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in importance],
        "curves": curves,
    }

    for key in model_probs:
        metrics.setdefault(key, {"roc_auc": curves[key]["roc_auc"], "pr_auc": curves[key]["pr_auc"]})
    metrics["comparison"] = comparison

    # 6. Артефакти
    out = Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fe, out / "feature_engineer.joblib")
    joblib.dump(iso, out / "iso_forest.joblib")
    joblib.dump(xgb, out / "xgboost_model.joblib")
    for key, fname in BASELINE_FILES.items():  # моделі порівняння: для оцінки якості на пакетах
        if key in fitted:
            joblib.dump(fitted[key], out / fname, compress=3)
        elif (out / fname).exists():
            (out / fname).unlink()  # застарілий файл
    config = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": list(X_train.columns),
        "thresholds": {**thresholds, "anomaly_review": anomaly_review_threshold},
        "costs": {"false_negative": COST_FN, "false_positive": COST_FP},
        "xgb_best_iteration": int(xgb.best_iteration),
        "smote": use_smote,
        "model_thresholds": {r["key"]: r["threshold"] for r in comparison},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    summary = {k: v for k, v in metrics.items() if k not in ("curves", "feature_importance")}
    print("Моделі навчені та збережені у", out)
    print("\nПорівняння моделей (тест, поріг = max F1 на валідації):")
    print(f"{'Модель':<50}{'Precision':>10}{'Recall':>8}{'F1':>8}{'ROC-AUC':>9}{'PR-AUC':>8}")
    for r in comparison:
        print(f"{r['name']:<50}{r['precision']:>10.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}{r['roc_auc']:>9.3f}{r['pr_auc']:>8.3f}")
    print()
    print(json.dumps({"thresholds": config["thresholds"], "metrics": summary}, indent=2, ensure_ascii=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw_transactions.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-smote", action="store_true")
    args = parser.parse_args()
    train_pipeline(args.data, args.models_dir, seed=args.seed, use_smote=not args.no_smote)


if __name__ == "__main__":
    main()