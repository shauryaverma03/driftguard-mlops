"""
End-to-End Pipeline Tests for DriftGuard.
Tests drift detection, simulation, retraining, validation gate, hot-swap, and metrics.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_drift_status_endpoint(client):
    response = client.get("/drift/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "drift_detected" in data
    assert "features" in data
    assert "p_value_threshold" in data


def test_traffic_simulation_and_drift_detection(client):
    # 1. Reset first
    client.post("/pipeline/reset")
    
    # 2. Inject drifted batch
    drift_resp = client.post("/simulate/drift?batch_size=40")
    assert drift_resp.status_code == 200
    d_data = drift_resp.json()
    assert d_data["status"] == "DRIFT_INJECTED"
    assert d_data["injected_count"] == 40

    # 3. Check drift status after injection
    status_resp = client.get("/drift/status")
    assert status_resp.status_code == 200
    s_data = status_resp.json()
    assert s_data["current_samples"] >= 40


def test_retrain_trigger_and_hot_swap(client):
    # Trigger manual retraining
    retrain_resp = client.post("/retrain/trigger")
    assert retrain_resp.status_code == 200
    r_data = retrain_resp.json()
    assert r_data["success"] is True
    assert r_data["status"] == "PROMOTED"
    assert "promoted_version" in r_data

    # Verify active model is hot-swapped
    model_resp = client.get("/models/active")
    assert model_resp.status_code == 200
    m_data = model_resp.json()
    assert m_data["version"] == r_data["promoted_version"]


def test_prometheus_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "driftguard_requests_total" in content
    assert "driftguard_feature_ks_pvalue" in content
    assert "driftguard_system_drift_detected" in content


def test_before_after_comparison_endpoint(client):
    response = client.get("/drift/comparison")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "old_accuracy_pct" in data
    assert "new_accuracy_pct" in data


def test_pipeline_events_log(client):
    response = client.get("/events")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert len(data["events"]) > 0
