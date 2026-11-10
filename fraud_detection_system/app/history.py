"""Історія перевірок користувача (SQLite): підсумки, повна схема transactions / risk_assessments / model_logs
і журнал заблокованих транзакцій зі зняттям блоку.

Відповідність типів ER-діаграми (PostgreSQL → SQLite):
    UUID, TIMESTAMPTZ (ISO-8601, UTC), JSONB → TEXT;  BOOLEAN → INTEGER (0/1);  FLOAT → REAL.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DB_FILE = Path(os.getenv("HISTORY_DB", "data/history.db"))
MAX_BLOCKED_PER_CHECK = 500  # скільки заблокованих рядків показувати для зняття блоку в кожній перевірці

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    created_at TEXT NOT NULL,
    label TEXT NOT NULL,
    source TEXT NOT NULL,
    rows INTEGER NOT NULL,
    n_allow INTEGER NOT NULL, n_review INTEGER NOT NULL, n_block INTEGER NOT NULL,
    amt_allow REAL NOT NULL, amt_review REAL NOT NULL, amt_block REAL NOT NULL,
    mean_prob REAL NOT NULL,
    review_th REAL, block_th REAL
);
CREATE INDEX IF NOT EXISTS ix_checks_user ON checks(username, created_at DESC);

-- 1) transactions: кожна перевірена транзакція (check_id і time_sec додано для зв'язку з історією перевірок)
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,                         -- UUID
    check_id INTEGER NOT NULL REFERENCES checks(id) ON DELETE CASCADE,
    user_id VARCHAR(64) NOT NULL,
    amount NUMERIC(12,2) NOT NULL,
    timestamp TEXT NOT NULL,                                 -- TIMESTAMPTZ: момент перевірки (UTC)
    time_sec FLOAT,                                          -- колонка Time з даних (секунди від початку спостереження)
    log_amount FLOAT NOT NULL,
    anomaly_score FLOAT,
    fraud_probability FLOAT NOT NULL,
    action_taken VARCHAR(16) NOT NULL,                       -- ALLOW / REVIEW / BLOCK / RELEASED (блок знято)
    is_fraud_actual INTEGER CHECK (is_fraud_actual IN (0, 1)),  -- BOOLEAN, з колонки Class, якщо вона є
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tx_check ON transactions(check_id, fraud_probability DESC);
CREATE INDEX IF NOT EXISTS ix_tx_user ON transactions(user_id, created_at DESC);

-- 2) risk_assessments: оцінки ризику моделлю (1 транзакція : N оцінок)
CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id TEXT PRIMARY KEY,                          -- UUID
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    anomaly_score FLOAT,
    fraud_probability FLOAT NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    assessed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ra_tx ON risk_assessments(transaction_id);

-- 3) model_logs: журнал викликів моделей (1 транзакція : N записів)
CREATE TABLE IF NOT EXISTS model_logs (
    log_id TEXT PRIMARY KEY,                                 -- UUID
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    model_name VARCHAR(50) NOT NULL,
    execution_time FLOAT,                                    -- секунди (для масиву: середній час на транзакцію)
    result TEXT,                                             -- JSONB: відповідь моделі
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ml_tx ON model_logs(transaction_id);

CREATE TABLE IF NOT EXISTS blocked (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_id INTEGER NOT NULL REFERENCES checks(id) ON DELETE CASCADE,
    username TEXT NOT NULL,
    row_idx INTEGER NOT NULL,
    transaction_id TEXT REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    time REAL, amount REAL, prob REAL,
    status TEXT NOT NULL DEFAULT 'blocked',
    updated_at TEXT, updated_by TEXT
);
CREATE INDEX IF NOT EXISTS ix_blocked_check ON blocked(check_id);
"""


def _conn() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(_SCHEMA)
    # міграція: у базах, створених попередньою версією, у blocked немає transaction_id
    if "transaction_id" not in {r["name"] for r in con.execute("PRAGMA table_info(blocked)")}:
        con.execute("ALTER TABLE blocked ADD COLUMN transaction_id TEXT")
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(username: str) -> str:
    return re.sub(r"\s+", "", username).lower()


def _num(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def record_check(username: str, df: pd.DataFrame, *, label: str, source: str = "dataset",
                 review_th: float | None = None, block_th: float | None = None,
                 model_version: str = "hybrid", model_name: str = "hybrid",
                 elapsed: float | None = None) -> int:
    """Зберігає перевірку: підсумок, усі транзакції, їхні оцінки ризику та журнал моделі. Повертає id перевірки.

    df: Time, Amount, fraud_probability, action; за наявності anomaly_score і Class (0/1).
    elapsed: загальний час оцінки в секундах (для model_logs.execution_time береться середній на транзакцію).
    """
    username = _norm(username)
    d = df.reset_index(drop=True)
    n_rows = len(d)
    act = d["action"]
    n = {a: int((act == a).sum()) for a in ("ALLOW", "REVIEW", "BLOCK")}
    s = {a: float(d.loc[act == a, "Amount"].sum()) for a in ("ALLOW", "REVIEW", "BLOCK")}
    now = _now()

    amount = d["Amount"].astype(float).to_numpy()
    prob = d["fraud_probability"].astype(float).to_numpy()
    anomaly = (pd.to_numeric(d["anomaly_score"], errors="coerce").to_numpy()
               if "anomaly_score" in d.columns else np.full(n_rows, np.nan))
    time_sec = d["Time"].astype(float).to_numpy() if "Time" in d.columns else np.full(n_rows, np.nan)
    actual = [None] * n_rows
    if "Class" in d.columns:
        labels = pd.to_numeric(d["Class"], errors="coerce").to_numpy()
        actual = [int(v) if v in (0, 1) else None for v in labels]
    log_amount = np.log1p(amount)
    per_tx_time = (elapsed / n_rows) if elapsed is not None and n_rows else None
    ids = [str(uuid.uuid4()) for _ in range(n_rows)]

    with _conn() as con:
        cid = int(con.execute(
            "INSERT INTO checks (username, created_at, label, source, rows, n_allow, n_review, n_block,"
            " amt_allow, amt_review, amt_block, mean_prob, review_th, block_th)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (username, now, label, source, n_rows, n["ALLOW"], n["REVIEW"], n["BLOCK"],
             s["ALLOW"], s["REVIEW"], s["BLOCK"], float(prob.mean()) if n_rows else 0.0, review_th, block_th),
        ).lastrowid)

        con.executemany(
            "INSERT INTO transactions (transaction_id, check_id, user_id, amount, timestamp, time_sec, log_amount,"
            " anomaly_score, fraud_probability, action_taken, is_fraud_actual, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [(ids[i], cid, username, round(amount[i], 2), now, _num(time_sec[i]), float(log_amount[i]),
              _num(anomaly[i]), float(prob[i]), str(act.iloc[i]), actual[i], now) for i in range(n_rows)])
        con.executemany(
            "INSERT INTO risk_assessments (assessment_id, transaction_id, anomaly_score, fraud_probability,"
            " model_version, assessed_at) VALUES (?,?,?,?,?,?)",
            [(str(uuid.uuid4()), ids[i], _num(anomaly[i]), float(prob[i]), model_version, now) for i in range(n_rows)])
        con.executemany(
            "INSERT INTO model_logs (log_id, transaction_id, model_name, execution_time, result, created_at)"
            " VALUES (?,?,?,?,?,?)",
            [(str(uuid.uuid4()), ids[i], model_name, per_tx_time,
              json.dumps({"fraud_probability": round(float(prob[i]), 6), "anomaly_score": _num(anomaly[i]),
                          "action": str(act.iloc[i])}, ensure_ascii=False), now) for i in range(n_rows)])

        top = d[act == "BLOCK"].sort_values("fraud_probability", ascending=False).head(MAX_BLOCKED_PER_CHECK)
        con.executemany(
            "INSERT INTO blocked (check_id, username, row_idx, transaction_id, time, amount, prob)"
            " VALUES (?,?,?,?,?,?,?)",
            [(cid, username, int(i), ids[int(i)], _num(time_sec[int(i)]), float(amount[int(i)]), float(prob[int(i)]))
             for i in top.index])
    return cid


def list_checks(username: str, limit: int = 200) -> pd.DataFrame:
    with _conn() as con:
        rows = con.execute(
            "SELECT c.*, (SELECT COUNT(*) FROM blocked b WHERE b.check_id = c.id AND b.status = 'released') AS n_released"
            " FROM checks c WHERE c.username = ? ORDER BY c.created_at DESC, c.id DESC LIMIT ?",
            (_norm(username), limit),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def list_blocked(username: str, check_id: int) -> pd.DataFrame:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, row_idx, time, amount, prob, status, updated_at FROM blocked"
            " WHERE check_id = ? AND username = ? ORDER BY prob DESC",
            (check_id, _norm(username)),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def list_transactions(username: str, check_id: int, limit: int = 1000) -> pd.DataFrame:
    """Транзакції перевірки з оцінками ризику, найризикованіші першими."""
    with _conn() as con:
        rows = con.execute(
            "SELECT t.transaction_id, t.time_sec, t.amount, t.log_amount, t.anomaly_score, t.fraud_probability,"
            " t.action_taken, t.is_fraud_actual, r.model_version FROM transactions t"
            " LEFT JOIN risk_assessments r ON r.transaction_id = t.transaction_id"
            " WHERE t.check_id = ? AND t.user_id = ? ORDER BY t.fraud_probability DESC LIMIT ?",
            (int(check_id), _norm(username), limit),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def set_blocked_status(username: str, blocked_ids: list[int], status: str) -> int:
    """status: 'released' (блок знято) або 'blocked' (блок повернуто). Діє лише на записи цього користувача.
    Дзеркально оновлює transactions.action_taken (RELEASED / BLOCK)."""
    if status not in ("released", "blocked"):
        raise ValueError("Невідомий статус")
    if not blocked_ids:
        return 0
    user, ids = _norm(username), [int(i) for i in blocked_ids]
    marks = ",".join("?" * len(ids))
    with _conn() as con:
        cur = con.execute(
            f"UPDATE blocked SET status = ?, updated_at = ?, updated_by = ? WHERE username = ? AND id IN ({marks})",
            (status, _now(), user, user, *ids),
        )
        con.execute(
            f"UPDATE transactions SET action_taken = ? WHERE user_id = ? AND transaction_id IN"
            f" (SELECT transaction_id FROM blocked WHERE username = ? AND id IN ({marks}))",
            ("RELEASED" if status == "released" else "BLOCK", user, user, *ids),
        )
    return cur.rowcount


def delete_check(username: str, check_id: int) -> None:
    with _conn() as con:
        con.execute("DELETE FROM checks WHERE id = ? AND username = ?", (int(check_id), _norm(username)))


def clear_history(username: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM checks WHERE username = ?", (_norm(username),))