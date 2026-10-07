"""Stage 4: train both candidates, track them in MLflow, package model-v1.

Run:  python -m src.train

Candidates (feature order always [G1, G2, studytime], fitted on TRAIN only):
  ridge          Pipeline(StandardScaler, Ridge(alpha=1.0))
  random_forest  Pipeline(RandomForestRegressor(n_estimators=200, max_depth=5,
                          min_samples_leaf=3, random_state=42, n_jobs=1))
No hyperparameter search. The test partition is never read here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    ACTIVE_MODEL_PATH, ARTIFACTS_DIR, DATA_DIR, FEATURES, GRADE_MAX, GRADE_MIN,
    RAW_CSV, REPORTS_DIR, SEED, TARGET, configure_mlflow, git_commit,
    load_partition, package_versions, predict_clipped, python_version,
    regression_metrics, sha256_file, utc_now, write_json,
)
from .report_text import render_model_comparison

TIE_TOLERANCE = 0.05  # grade points


def build_candidates() -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=200, max_depth=5, min_samples_leaf=3,
                random_state=SEED, n_jobs=1)),
        ]),
    }


def training_parameters(pipeline: Pipeline) -> dict:
    """Flat, JSON-safe copy of the estimator parameters."""
    params = {}
    for key, value in pipeline.get_params(deep=True).items():
        if "__" not in key:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            params[key] = value
    return params


def choose_candidate(results: dict[str, dict]) -> tuple[str, float]:
    """Lower validation MAE wins; within 0.05 grade points prefer Ridge."""
    ridge_mae = results["ridge"]["metrics"]["mae"]
    forest_mae = results["random_forest"]["metrics"]["mae"]
    difference = abs(ridge_mae - forest_mae)
    if difference <= TIE_TOLERANCE:
        return "ridge", difference
    return ("ridge" if ridge_mae < forest_mae else "random_forest"), difference


def fit_and_score(mlflow, name: str, pipeline: Pipeline, train: pd.DataFrame, validation: pd.DataFrame,
                  dataset_sha: str, split_sha: str, commit: str, out_dir: Path) -> dict:
    pipeline.fit(train[FEATURES], train[TARGET])
    predictions = predict_clipped(pipeline, validation)
    metrics = regression_metrics(validation[TARGET], predictions)
    candidate_path = out_dir / f"candidate-{name}.joblib"
    joblib.dump(pipeline, candidate_path)

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(training_parameters(pipeline))
        mlflow.log_param("feature_order", ",".join(FEATURES))
        mlflow.log_param("clipping", f"[{GRADE_MIN},{GRADE_MAX}]")
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("validation_rows", len(validation))
        mlflow.set_tags({"dataset_sha256": dataset_sha, "split_manifest_sha256": split_sha,
                         "git_commit": commit, "stage": "candidate-training"})
        mlflow.log_metrics({"validation_mae": metrics["mae"],
                            "validation_rmse": metrics["rmse"],
                            "validation_r2": metrics["r2"]})
        mlflow.log_artifact(str(candidate_path))
        run_id = run.info.run_id
    return {"pipeline": pipeline, "metrics": metrics, "run_id": run_id, "path": candidate_path}


def main() -> int:
    train = load_partition("train")
    validation = load_partition("validation")
    baselines_path = REPORTS_DIR / "baselines.csv"
    if not baselines_path.exists():
        print("ERROR: run `python -m src.baselines` first.")
        return 1
    baselines = pd.read_csv(baselines_path)

    splits_path = DATA_DIR / "splits.json"
    dataset_sha = sha256_file(RAW_CSV) if RAW_CSV.exists() else "raw file not present on this machine"
    split_sha = sha256_file(splits_path)
    commit = git_commit()

    mlflow = configure_mlflow()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = ARTIFACTS_DIR / "candidates"
    work_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, pipeline in build_candidates().items():
        results[name] = fit_and_score(mlflow, name, pipeline, train, validation,
                                      dataset_sha, split_sha, commit, work_dir)
        m = results[name]["metrics"]
        print(f"{name:<14} MAE {m['mae']:.4f}  RMSE {m['rmse']:.4f}  R2 {m['r2']:.4f}  run {results[name]['run_id']}")

    chosen, difference = choose_candidate(results)
    chosen_info = results[chosen]
    print(f"Selected: {chosen} (MAE difference between candidates {difference:.4f})")

    # Save v1 (already fitted on train only; never re-fitted on validation/test).
    artifact = ARTIFACTS_DIR / "model-v1.joblib"
    joblib.dump(chosen_info["pipeline"], artifact)

    metadata = {
        "model_version": "v1",
        "model_family": chosen,
        "created_at_utc": utc_now(),
        "feature_names": FEATURES,
        "target_name": TARGET,
        "grade_range": [GRADE_MIN, GRADE_MAX],
        "clipping_enabled": True,
        "training_parameters": training_parameters(chosen_info["pipeline"]),
        "dataset_sha256": dataset_sha,
        "split_manifest_sha256": split_sha,
        "git_commit": commit,
        "python_version": python_version(),
        "package_versions": package_versions(),
        "validation_mae": chosen_info["metrics"]["mae"],
        "validation_rmse": chosen_info["metrics"]["rmse"],
        "validation_r2": chosen_info["metrics"]["r2"],
        "mlflow_run_id": chosen_info["run_id"],
    }
    write_json(ARTIFACTS_DIR / "model-v1.json", metadata)
    write_json(ACTIVE_MODEL_PATH, {"model_version": "v1", "artifact_path": "artifacts/model-v1.joblib"})

    # Verify reload: saved and reloaded predictions must agree within 1e-6.
    reloaded = joblib.load(artifact)
    sample = validation.head(20)
    diff = float(np.max(np.abs(predict_clipped(chosen_info["pipeline"], sample)
                               - predict_clipped(reloaded, sample))))
    if diff > 1e-6:
        print(f"ERROR: reloaded model differs by {diff}")
        return 1
    print(f"Reload check OK (max difference {diff:.2e})")

    comparison = {"results": {k: {"metrics": v["metrics"], "run_id": v["run_id"]} for k, v in results.items()},
                  "chosen": chosen, "mae_difference": difference}
    (REPORTS_DIR / "model-comparison.md").write_text(
        render_model_comparison(baselines, comparison, generated_at=utc_now()), encoding="utf-8")
    print("OK: wrote artifacts/model-v1.joblib, model-v1.json, active-model.json, reports/model-comparison.md")
    print("Open MLflow (see README for the env variable):  python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
