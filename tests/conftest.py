"""Shared test fixtures. Tests use tiny synthetic data and a tiny fitted pipeline:
they never need the real dataset, a trained release artifact, MLflow or a running server.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import FEATURES


def synthetic_frame(rows: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g1 = rng.integers(0, 21, rows)
    g2 = np.clip(g1 + rng.integers(-2, 3, rows), 0, 20)
    studytime = rng.integers(1, 5, rows)
    g3 = np.clip(g2 + rng.integers(-2, 3, rows), 0, 20)
    return pd.DataFrame({"source_row_id": range(rows), "G1": g1, "G2": g2,
                         "studytime": studytime, "G3": g3})


@pytest.fixture
def frame():
    return synthetic_frame()


@pytest.fixture
def tiny_pipeline():
    data = synthetic_frame(120, seed=1)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipe.fit(data[FEATURES], data["G3"])
    return pipe


@pytest.fixture
def active_manifest(tmp_path, tiny_pipeline):
    """A temporary artifact + active-model.json, as the API expects."""
    artifact = tmp_path / "model-v1.joblib"
    joblib.dump(tiny_pipeline, artifact)
    manifest = tmp_path / "active-model.json"
    manifest.write_text(json.dumps({"model_version": "v1", "artifact_path": str(artifact)}))
    return manifest
