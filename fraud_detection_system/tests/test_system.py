"""Інтеграційні тести:  ."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from src.feature_engineering import FEATURE_COLUMNS, RAW_COLUMNS, TransactionFeatureEngineer
from src.generate_data import generate
from src.pipeline import FraudPipeline
from src.train import oof_isolation_scores, train_pipeline


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("fraud")
    data = root / "data.csv"
    generate(n_rows=8000, fraud_rate=0.03, seed=1).to_csv(data, index=False)
    metrics = train_pipeline(str(data), str(root / "models"), xgb_n_estimators=60)
    return root / "models", data, metrics


def test_feature_engineering_columns_and_no_nan():
    df = generate(n_rows=500, fraud_rate=0.05, seed=2)
    fe = TransactionFeatureEngineer()
    X = fe.fit_transform(df[RAW_COLUMNS])
    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any().any()
    assert set(X["is_night"].unique()) <= {0.0, 1.0}


def test_feature_engineering_rejects_bad_input():
    fe = TransactionFeatureEngineer()
    with pytest.raises(ValueError):
        fe.transform(pd.DataFrame({"Time": [1.0]}))


def test_oof_scores_cover_every_row():
    X = TransactionFeatureEngineer().fit_transform(generate(600, 0.05, 3)[RAW_COLUMNS])
    scores = oof_isolation_scores(X, seed=0, n_splits=3)
    assert scores.shape == (len(X),) and np.all(scores != 0)


def test_model_quality(trained):
    _, _, metrics = trained
    assert metrics["hybrid"]["pr_auc"] > 0.7
    assert metrics["hybrid"]["roc_auc"] > 0.9


def test_pipeline_predicts_fraud_higher_than_legit(trained):
    models_dir, data, _ = trained
    df = pd.read_csv(data)
    pipe = FraudPipeline.load(models_dir)
    res, expl = pipe.predict(df.head(300), explain_top_k=3)
    assert set(res["action"]) <= {"ALLOW", "REVIEW", "BLOCK"}
    assert len(expl) == 300 and len(expl[0]) == 3
    labels = df.head(300)["Class"]
    assert res.loc[labels == 1, "fraud_probability"].mean() > res.loc[labels == 0, "fraud_probability"].mean()


@pytest.fixture
def client(trained):
    with TestClient(create_app(str(trained[0]))) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_minimal_payload_and_explain(client):
    r = client.post("/predict", params={"explain": True, "top_k": 4}, json={"Time": 3600, "Amount": 10.0})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["action"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert len(body["explanation"]) == 4
    assert "X-Process-Time-ms" in r.headers


def test_predict_validation(client):
    assert client.post("/predict", json={"Time": 1, "Amount": -5}).status_code == 422
    assert client.post("/predict", json={"Time": 1}).status_code == 422
    assert client.post("/predict", json={"Time": 1, "Amount": 1, "v1": 0.3}).status_code == 422


def test_batch(client, trained):
    df = pd.read_csv(trained[1]).head(20)
    rows = df[RAW_COLUMNS].to_dict("records")
    r = client.post("/predict/batch", json={"transactions": rows})
    assert r.status_code == 200 and len(r.json()["results"]) == 20
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_missing_models_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        FraudPipeline.load(tmp_path)
