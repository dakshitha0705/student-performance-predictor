"""Prediction API.

  GET  /health   {"status":"ready","model_version":"v1"} only after the model loaded
  POST /predict  {"G1":12,"G2":14,"studytime":2} -> prediction, scale, model_version

Rules from the guide (section 11/12):
  * strict integer validation; booleans and unknown fields are rejected (422)
  * the model is loaded ONCE at startup from artifacts/active-model.json
  * if loading fails the service is unready: /health and /predict return 503
  * nothing is fitted here; a request can never choose a file to load
  * one JSON line per /predict request (success, validation_error, service_error);
    rejected requests never have their raw body written to the log
  * run with ONE worker (log file writes are not coordinated across processes)

Run:  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.common import ACTIVE_MODEL_PATH, FEATURES, ROOT, load_active_model, predict_clipped

DEFAULT_LOG_PATH = ROOT / "logs" / "predictions.jsonl"


class PredictRequest(BaseModel):
    # strict=True: "12", 12.0 and True are all rejected; extra="forbid": unknown fields rejected.
    model_config = ConfigDict(extra="forbid", strict=True)

    G1: Annotated[int, Field(strict=True, ge=0, le=20)]
    G2: Annotated[int, Field(strict=True, ge=0, le=20)]
    studytime: Annotated[int, Field(strict=True, ge=1, le=4)]


class PredictResponse(BaseModel):
    prediction: float
    scale: str = "0-20"
    model_version: str


class EventLogger:
    """Appends one JSON object per line. Thread-safe within a single process."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, event: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(event, ensure_ascii=False)
            with self._lock, open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            # Never let a logging problem break a prediction.
            pass


def create_app(active_model_path: Path | str | None = None,
               log_path: Path | str | None = None) -> FastAPI:
    active_path = Path(active_model_path or os.environ.get("ACTIVE_MODEL_PATH", ACTIVE_MODEL_PATH))
    logger = EventLogger(log_path or os.environ.get("LOG_PATH", DEFAULT_LOG_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.model_version = None
        app.state.load_error = None
        try:
            app.state.model, app.state.model_version = load_active_model(active_path)
        except Exception as exc:  # unreadable manifest, missing artifact, bad pickle ...
            app.state.load_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="Student Performance Predictor API", lifespan=lifespan)

    @app.middleware("http")
    async def log_predictions(request: Request, call_next):
        if not (request.method == "POST" and request.url.path == "/predict"):
            return await call_next(request)

        request_id = uuid.uuid4().hex
        request.state.event_extra = {}
        started = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            response = JSONResponse({"detail": "Internal server error"}, status_code=500)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        if status_code == 200:
            status = "success"
        elif status_code == 422:
            status = "validation_error"
        else:
            status = "service_error"
        event = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "request_id": request_id,
            "status": status,
            "latency_ms": latency_ms,
            "model_version": getattr(request.app.state, "model_version", None),
        }
        if status == "success":
            event.update(request.state.event_extra)  # inputs + prediction only
        logger.write(event)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Report field, location and reason only; do not echo or log the raw input.
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health")
    async def health(request: Request):
        if request.app.state.model is None:
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return {"status": "ready", "model_version": request.app.state.model_version}

    @app.post("/predict", response_model=PredictResponse)
    async def predict(payload: PredictRequest, request: Request):
        if request.app.state.model is None:
            return JSONResponse(status_code=503, content={"detail": "Model is not available"})
        row = pd.DataFrame([[payload.G1, payload.G2, payload.studytime]], columns=FEATURES)
        value = float(predict_clipped(request.app.state.model, row)[0])
        request.state.event_extra = {
            "inputs": {"G1": payload.G1, "G2": payload.G2, "studytime": payload.studytime},
            "prediction": value,
        }
        return PredictResponse(prediction=value, model_version=request.app.state.model_version)

    return app


app = create_app()
