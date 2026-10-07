"""Shared constants and helpers used by every stage of the pipeline.

Keep this file small: anything that must be identical in training, evaluation
and serving (feature order, clipping, metrics) lives here so it cannot drift.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "student-mat.csv"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
ACTIVE_MODEL_PATH = ARTIFACTS_DIR / "active-model.json"

# The feature order is part of the model contract. Never reorder.
FEATURES = ["G1", "G2", "studytime"]
TARGET = "G3"
GRADE_MIN, GRADE_MAX = 0, 20
SEED = 42
MLFLOW_EXPERIMENT = "student-performance"

PACKAGES = [
    "pandas", "numpy", "scikit-learn", "mlflow", "fastapi", "uvicorn",
    "streamlit", "requests", "pytest", "httpx", "joblib", "matplotlib",
]


def configure_mlflow():
    """Point MLflow at the local file store in mlruns/ (guide section 10).

    Recent MLflow releases refuse the file store unless explicitly allowed, so the
    opt-in variable is set here before any tracking call. The same variable must be
    set when starting the UI (see README).
    """
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return mlflow


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def python_version() -> str:
    return f"{platform.python_version()} ({sys.platform})"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=False)
        handle.write("\n")


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_partition(name: str) -> pd.DataFrame:
    """Load train / validation / test / later from data/processed."""
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.validate` and "
            f"`python -m src.split` first."
        )
    return pd.read_csv(path)


def predict_clipped(model, frame: pd.DataFrame) -> np.ndarray:
    """The single prediction helper used in evaluation AND serving.

    Selects the features in the agreed order and clips to [0, 20]. Values are
    NOT rounded here; rounding is for display only.
    """
    ordered = frame[FEATURES]
    return np.clip(np.asarray(model.predict(ordered), dtype=float), GRADE_MIN, GRADE_MAX)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def resolve_artifact_path(artifact_path: str) -> Path:
    path = Path(artifact_path)
    return path if path.is_absolute() else ROOT / path


def load_active_model(active_path: Path | None = None):
    """Load the model named in active-model.json.

    Returns (pipeline, model_version). Only the path written in the manifest is
    ever opened; nothing here is influenced by a request.
    """
    import joblib

    manifest_path = Path(active_path) if active_path else ACTIVE_MODEL_PATH
    manifest = read_json(manifest_path)
    model = joblib.load(resolve_artifact_path(manifest["artifact_path"]))
    return model, manifest["model_version"]
