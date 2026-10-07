"""Drives the Streamlit form in-process with a fake API. No server needed."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def fake_response(status, body=None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body or {}
    return response


def texts(at):
    return [m.value for m in at.markdown] + [e.value for e in at.error] + [s.value for s in at.success] + \
           [c.value for c in at.caption]


def test_defaults_and_title():
    at = AppTest.from_file(APP).run()
    assert at.title[0].value == "Student Performance Predictor"
    assert at.number_input[0].value == 12
    assert at.number_input[1].value == 14
    assert at.selectbox[0].value == "2 to 5 hours"
    assert at.selectbox[0].options == ["Less than 2 hours", "2 to 5 hours", "5 to 10 hours", "More than 10 hours"]
    assert at.button[0].label == "Predict final grade"


def test_success_shows_rounded_value_version_and_limitation():
    at = AppTest.from_file(APP).run()
    body = {"prediction": 12.2192, "scale": "0-20", "model_version": "v1"}
    with patch("requests.post", return_value=fake_response(200, body)) as post:
        at.button[0].click().run()
    sent = post.call_args.kwargs["json"]
    assert sent == {"G1": 12, "G2": 14, "studytime": 2}
    assert post.call_args.kwargs["timeout"] == 10
    assert at.success[0].value == "Estimated final grade: 12.2 / 20"
    assert any("Model version: v1" in t for t in texts(at))
    assert any("not a guaranteed result" in t for t in texts(at))


def test_study_choice_maps_to_numeric_category():
    at = AppTest.from_file(APP).run()
    at.selectbox[0].select("More than 10 hours")
    with patch("requests.post", return_value=fake_response(200, {"prediction": 15.0, "model_version": "v1"})) as post:
        at.button[0].click().run()
    assert post.call_args.kwargs["json"]["studytime"] == 4


def test_connection_error_message_and_no_stale_result():
    at = AppTest.from_file(APP).run()
    with patch("requests.post", return_value=fake_response(200, {"prediction": 15.0, "model_version": "v1"})):
        at.button[0].click().run()
    assert len(at.success) == 1
    with patch("requests.post", side_effect=requests.ConnectionError()):
        at.button[0].click().run()
    assert len(at.success) == 0                      # previous success was cleared
    assert at.error[0].value == "Prediction service is unavailable. Check that the API is running and try again."


def test_422_shows_a_validation_error():
    at = AppTest.from_file(APP).run()
    with patch("requests.post", return_value=fake_response(422, {"detail": []})):
        at.button[0].click().run()
    assert len(at.success) == 0
    assert "rejected" in at.error[0].value
