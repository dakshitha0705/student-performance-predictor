"""Test API logging without loading the real trained model."""

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import api.main as api_module


VALID_INPUT = {
    "G1": 12,
    "G2": 14,
    "studytime": 2,
}


def read_events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def client_and_log(tmp_path, monkeypatch):
    # Substitute predictable model behaviour for these logging tests.
    monkeypatch.setattr(
        api_module,
        "load_active_model",
        lambda path: (object(), "test-v1"),
    )
    monkeypatch.setattr(
        api_module,
        "predict_clipped",
        lambda model, row: [13.5],
    )

    log_path = tmp_path / "predictions.jsonl"

    app = api_module.create_app(
        active_model_path=tmp_path / "unused-model.json",
        log_path=log_path,
    )

    with TestClient(app) as client:
        yield client, log_path


def test_success_logs_validated_inputs_and_prediction(client_and_log):
    client, log_path = client_and_log

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200

    events = read_events(log_path)
    assert len(events) == 1

    event = events[0]
    assert event["status"] == "success"
    assert event["inputs"] == VALID_INPUT
    assert event["prediction"] == response.json()["prediction"]
    assert event["model_version"] == "test-v1"


def test_request_ids_are_unique(client_and_log):
    client, log_path = client_and_log

    client.post("/predict", json=VALID_INPUT)
    client.post("/predict", json=VALID_INPUT)

    events = read_events(log_path)

    assert len(events) == 2
    assert all(event["request_id"] for event in events)
    assert events[0]["request_id"] != events[1]["request_id"]


def test_timestamp_is_utc_and_latency_is_nonnegative(client_and_log):
    client, log_path = client_and_log

    client.post("/predict", json=VALID_INPUT)

    event = read_events(log_path)[0]

    timestamp = datetime.fromisoformat(
        event["timestamp_utc"].replace("Z", "+00:00")
    )

    assert timestamp.utcoffset().total_seconds() == 0
    assert event["latency_ms"] >= 0


@pytest.mark.parametrize(
    "payload",
    [
        {"G1": -1, "G2": 14, "studytime": 2},
        {"G1": 12, "studytime": 2},
        {"G1": True, "G2": 14, "studytime": 2},
        {"G1": 12.5, "G2": 14, "studytime": 2},
        {
            "G1": 12,
            "G2": 14,
            "studytime": 2,
            "secret_note": "DO_NOT_LOG_THIS",
        },
    ],
)
def test_invalid_requests_do_not_log_payload(client_and_log, payload):
    client, log_path = client_and_log

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

    events = read_events(log_path)
    assert len(events) == 1

    event = events[0]
    assert event["status"] == "validation_error"

    assert set(event) == {
        "timestamp_utc",
        "request_id",
        "status",
        "latency_ms",
        "model_version",
    }

    assert "DO_NOT_LOG_THIS" not in log_path.read_text(
        encoding="utf-8"
    )


def test_missing_model_logs_service_error(tmp_path, monkeypatch):
    def fail_to_load(path):
        raise FileNotFoundError("Test model is unavailable")

    monkeypatch.setattr(
        api_module,
        "load_active_model",
        fail_to_load,
    )

    log_path = tmp_path / "predictions.jsonl"
    app = api_module.create_app(
        active_model_path=tmp_path / "missing.json",
        log_path=log_path,
    )

    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 503

    events = read_events(log_path)
    assert len(events) == 1

    event = events[0]
    assert event["status"] == "service_error"
    assert event["model_version"] is None
    assert "inputs" not in event
    assert "prediction" not in event


def test_prediction_failure_logs_service_error(
    client_and_log, monkeypatch
):
    client, log_path = client_and_log

    def fail_prediction(model, row):
        raise RuntimeError("DO_NOT_LOG_INTERNAL_DETAILS")

    monkeypatch.setattr(
        api_module,
        "predict_clipped",
        fail_prediction,
    )

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 500

    events = read_events(log_path)
    assert len(events) == 1
    assert events[0]["status"] == "service_error"
    assert "inputs" not in events[0]
    assert "prediction" not in events[0]
    assert "DO_NOT_LOG_INTERNAL_DETAILS" not in (
        log_path.read_text(encoding="utf-8")
    )