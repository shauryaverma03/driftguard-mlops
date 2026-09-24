"""
Integration tests for FastAPI Serving & Model Inference.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.model_manager import model_manager
from backend.config import settings


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "active_model_version" in data


def test_predict_manipulation_trade(client):
    # Simulated spoofing/layering trade
    payload = {
        "cancel_ratio": 0.95,
        "order_to_cancel_latency_ms": 10.0,
        "order_size_zscore": 4.5,
        "order_frequency": 160.0,
        "time_between_orders_ms": 15.0,
        "ground_truth": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "is_manipulation" in data
    assert data["model_version"] is not None
    assert data["request_id"] > 0


def test_predict_normal_trade(client):
    # Simulated legitimate retail trade
    payload = {
        "cancel_ratio": 0.05,
        "order_to_cancel_latency_ms": 1500.0,
        "order_size_zscore": -0.2,
        "order_frequency": 5.0,
        "time_between_orders_ms": 1800.0,
        "ground_truth": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["is_manipulation"] is False
