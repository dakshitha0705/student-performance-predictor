import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client(active_manifest, tmp_path):
    log = tmp_path / "predictions.jsonl"
    app = create_app(active_model_path=active_manifest, log_path=log)
    with TestClient(app) as test_client:
        test_client.log_path = log
        yield test_client


def events(client):
    if not client.log_path.exists():
        return []
    return [json.loads(line) for line in client.log_path.read_text().splitlines() if line.strip()]


def test_health_ready_with_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_version": "v1"}


def test_valid_prediction_has_contract_fields(client):
    response = client.post("/predict", json={"G1": 12, "G2": 14, "studytime": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["scale"] == "0-20"
    assert body["model_version"] == "v1"
    assert isinstance(body["prediction"], float)
    assert 0 <= body["prediction"] <= 20


@pytest.mark.parametrize("payload", [
    {"G1": -1, "G2": 14, "studytime": 2},          # below range
    {"G1": 21, "G2": 14, "studytime": 2},          # above range
    {"G1": 12, "G2": 14, "studytime": 0},          # studytime out of range
    {"G1": 12, "G2": 14, "studytime": 5},
    {"G1": 12, "G2": 14},                          # missing field
    {"G1": 12.5, "G2": 14, "studytime": 2},        # float
    {"G1": "12", "G2": 14, "studytime": 2},        # string
    {"G1": True, "G2": 14, "studytime": 2},        # boolean
    {"G1": 12, "G2": 14, "studytime": 2, "G3": 15},  # unknown field (G3 is never an input)
    {"G1": 12, "G2": 14, "studytime": 2, "name": "x"},
])
def test_invalid_payloads_return_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_unavailable_model_returns_503(tmp_path):
    app = create_app(active_model_path=tmp_path / "does-not-exist.json", log_path=tmp_path / "log.jsonl")
    with TestClient(app) as test_client:
        assert test_client.get("/health").status_code == 503
        assert test_client.post("/predict", json={"G1": 12, "G2": 14, "studytime": 2}).status_code == 503


def test_prediction_is_logged_with_inputs_and_version(client):
    client.post("/predict", json={"G1": 12, "G2": 14, "studytime": 2})
    logged = events(client)
    assert len(logged) == 1
    event = logged[0]
    assert event["status"] == "success"
    assert event["model_version"] == "v1"
    assert event["inputs"] == {"G1": 12, "G2": 14, "studytime": 2}
    assert isinstance(event["prediction"], float)
    assert event["latency_ms"] >= 0
    assert len(event["request_id"]) == 32
    assert event["timestamp_utc"].endswith("Z")


def test_rejected_requests_are_logged_without_raw_payload(client):
    client.post("/predict", json={"G1": -1, "G2": 14, "studytime": 2, "name": "Asha Rao"})
    logged = events(client)
    assert len(logged) == 1
    assert logged[0]["status"] == "validation_error"
    assert "inputs" not in logged[0] and "prediction" not in logged[0]
    assert "Asha" not in json.dumps(logged[0])


def test_service_error_is_logged(tmp_path):
    log = tmp_path / "log.jsonl"
    app = create_app(active_model_path=tmp_path / "missing.json", log_path=log)
    with TestClient(app) as test_client:
        test_client.post("/predict", json={"G1": 12, "G2": 14, "studytime": 2})
    event = json.loads(log.read_text().splitlines()[0])
    assert event["status"] == "service_error"
    assert event["model_version"] is None


def test_health_checks_are_not_logged(client):
    client.get("/health")
    assert events(client) == []
