# Fraud Detection System

Гібридна система виявлення аномалій і шахрайства в транзакціях:
**Isolation Forest** (аномальність без міток) → **XGBoost** (класифікація) → рішення `ALLOW / REVIEW / BLOCK`,
REST API на **FastAPI**, дашборд аналітика на **Streamlit**.

## Швидкий старт

```bash
pip install -r requirements-dev.txt

# 1. Дані: синтетичні (або покладіть реальний Kaggle creditcard.csv у data/raw_transactions.csv)
python -m src.generate_data --rows 100000 --fraud-rate 0.006

# 2. Навчання (артефакти та metrics.json потраплять у models/)
python -m src.train

# 3. Тести
pytest

# 4. API (http://localhost:8000/docs) та дашборд (http://localhost:8501)
uvicorn app.main:app --port 8000
streamlit run app/dashboard.py
```

Через Docker:

```bash
docker compose --profile train run --rm trainer   # генерує дані за потреби й навчає моделі
docker compose up --build                          # API :8000 + дашборд :8501
```

## Структура

```
src/feature_engineering.py   ознаки (час доби, log/Robust-масштабування суми), єдиний порядок колонок
src/generate_data.py         синтетичний датасет зі схемою Kaggle (Time, V1..V28, Amount, Class)
src/train.py                 навчання: OOF Isolation Forest, SMOTE, XGBoost, пороги, метрики
src/pipeline.py              FraudPipeline: ознаки → anomaly_score → ймовірність → рішення + пояснення
app/main.py                  FastAPI: /predict, /predict/batch, /health
app/dashboard.py             Streamlit: одна транзакція, пакетна CSV-перевірка, XAI-графік
tests/test_system.py         інтеграційні тести (навчання на малому наборі → пайплайн → API)
```

Артефакти в `models/`: `feature_engineer.joblib`, `iso_forest.joblib`, `xgboost_model.joblib`,
`config.json` (пороги, порядок ознак), `metrics.json` (якість на тесті).

## Що змінено порівняно з початковим описом

| Проблема в оригіналі | Виправлення |
|---|---|
| `anomaly_score` на train рахувала модель, що вже бачила ці точки | out-of-fold оцінки (`oof_isolation_scores`) |
| Масштабувальник навчався на всьому датасеті до split | split першим, `fit` лише на train |
| API приймав лише V1–V4, модель навчена на V1–V28 | усі V1–V28 у схемі (за замовчуванням 0.0), `extra="forbid"` |
| Порядок колонок міг розійтися між train і API | `FEATURE_COLUMNS` + `config.json` |
| Імпорт `from feature_engineering import ...` ламався поза `src/` | пакетні імпорти, запуск `python -m src.train` |
| `scaler.joblib` насправді містив весь `TransactionFeatureEngineer` | перейменовано на `feature_engineer.joblib` |
| Пороги 0.8 / 0.4 / -0.15 «з потолка» | підбір на валідації за вартістю помилок (`COST_FN`, `COST_FP` у `train.py`) |
| SHAP-заглушка в дашборді | реальні вклади ознак через нативний `pred_contribs` XGBoost (без залежності `shap`) |
| `tx.dict()`, `except Exception → 500 з текстом помилки` | `model_dump()`, `422` для некоректних даних |
| Тверження «< 30 мс» без вимірювань | заголовок `X-Process-Time-ms` у кожній відповіді API |

## Нотатки

- `anomaly_score` — це `IsolationForest.score_samples`: **менше значення = аномальніше**.
- Після SMOTE імовірності XGBoost завищені відносно реальної частки шахрайства, тому пороги підбираються
  на валідаційній вибірці **без** SMOTE.
- Якість оцінюйте за PR-AUC (`metrics.json`), а не за accuracy. Там же є абляція: гібрид проти XGBoost
  без `anomaly_score` та проти самого Isolation Forest.
- Синтетичні шахрайські транзакції навмисно «легкі», тож метрики на них завищені. Реальну якість покаже
  датасет Kaggle Credit Card Fraud (`data/raw_transactions.csv`).
