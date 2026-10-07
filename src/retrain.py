"""Stage 6: retrain on train + later, decide whether to promote, never auto-release.

Run:  python -m src.retrain

  1. Combine train.csv and later.csv; fit the SAME model family with unchanged parameters.
  2. Save artifacts/model-v2.joblib and artifacts/model-v2.json.
  3. Score v1 and v2 on the UNCHANGED validation set with the shared clipped helper.
  4. Promote v2 only if its validation MAE is at least 0.05 lower than v1 AND the
     test suite passes. Otherwise v1 stays and the rejection is documented.

This script never edits active-model.json. Switching is done explicitly with
`python -m src.release ...` so a rejected candidate cannot be released by accident.
"""
from __future__ import annotations

import subprocess
import sys

import joblib
import pandas as pd

from .common import (
    ARTIFACTS_DIR, DATA_DIR, FEATURES, GRADE_MAX, GRADE_MIN,
    RAW_CSV, REPORTS_DIR, ROOT, TARGET, configure_mlflow, git_commit, load_partition, package_versions,
    predict_clipped, python_version, read_json, regression_metrics, sha256_file,
    utc_now, write_json,
)
from .train import build_candidates, training_parameters

MIN_IMPROVEMENT = 0.05


def should_promote(v1_mae: float, v2_mae: float, tests_passed: bool) -> bool:
    """v2 must be at least 0.05 grade points better on validation MAE and tests must pass."""
    return tests_passed and (v1_mae - v2_mae) >= MIN_IMPROVEMENT


def run_tests() -> bool:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    return result.returncode == 0


def main() -> int:
    v1_meta_path = ARTIFACTS_DIR / "model-v1.json"
    if not v1_meta_path.exists():
        print("ERROR: model-v1 not found. Run `python -m src.train` first.")
        return 1
    v1_meta = read_json(v1_meta_path)
    family = v1_meta["model_family"]

    train = load_partition("train")
    later = load_partition("later")
    validation = load_partition("validation")
    combined = pd.concat([train, later], ignore_index=True)

    pipeline = build_candidates()[family]  # unchanged parameters, fresh estimator
    pipeline.fit(combined[FEATURES], combined[TARGET])
    v2_metrics = regression_metrics(validation[TARGET], predict_clipped(pipeline, validation))

    v1_model = joblib.load(ARTIFACTS_DIR / "model-v1.joblib")
    v1_metrics = regression_metrics(validation[TARGET], predict_clipped(v1_model, validation))

    artifact = ARTIFACTS_DIR / "model-v2.joblib"
    joblib.dump(pipeline, artifact)

    mlflow = configure_mlflow()
    splits_path = DATA_DIR / "splits.json"
    dataset_sha = sha256_file(RAW_CSV) if RAW_CSV.exists() else "raw file not present on this machine"
    with mlflow.start_run(run_name=f"retrain-v2-{family}") as run:
        mlflow.log_params(training_parameters(pipeline))
        mlflow.log_param("training_rows", len(combined))
        mlflow.log_param("source", "train.csv + later.csv")
        mlflow.log_metrics({"validation_mae": v2_metrics["mae"], "validation_rmse": v2_metrics["rmse"],
                            "validation_r2": v2_metrics["r2"]})
        mlflow.set_tags({"stage": "retraining", "git_commit": git_commit()})
        mlflow.log_artifact(str(artifact))
        run_id = run.info.run_id

    write_json(ARTIFACTS_DIR / "model-v2.json", {
        "model_version": "v2",
        "model_family": family,
        "created_at_utc": utc_now(),
        "feature_names": FEATURES,
        "target_name": TARGET,
        "grade_range": [GRADE_MIN, GRADE_MAX],
        "clipping_enabled": True,
        "training_parameters": training_parameters(pipeline),
        "training_data": "train.csv + later.csv",
        "dataset_sha256": dataset_sha,
        "split_manifest_sha256": sha256_file(splits_path),
        "git_commit": git_commit(),
        "python_version": python_version(),
        "package_versions": package_versions(),
        "validation_mae": v2_metrics["mae"],
        "validation_rmse": v2_metrics["rmse"],
        "validation_r2": v2_metrics["r2"],
        "mlflow_run_id": run_id,
    })

    print("Running the test suite before any promotion decision ...")
    tests_passed = run_tests()
    promote = should_promote(v1_metrics["mae"], v2_metrics["mae"], tests_passed)
    improvement = v1_metrics["mae"] - v2_metrics["mae"]

    decision = {
        "created_at_utc": utc_now(),
        "model_family": family,
        "v1_validation": v1_metrics,
        "v2_validation": v2_metrics,
        "mae_improvement_v1_minus_v2": improvement,
        "required_improvement": MIN_IMPROVEMENT,
        "tests_passed": tests_passed,
        "decision": "promote_v2" if promote else "keep_v1",
        "v2_mlflow_run_id": run_id,
        "note": "active-model.json was NOT changed by this script.",
    }
    write_json(REPORTS_DIR / "retraining-decision.json", decision)

    reason = ("v2 validation MAE is at least 0.05 grade points lower than v1 and all tests passed."
              if promote else
              "v2 did not meet the promotion rule (needs MAE at least 0.05 lower than v1 and passing tests); "
              "v1 is retained and v2 is recorded as a rejected candidate.")
    (REPORTS_DIR / "retraining-decision.md").write_text("\n".join([
        "# Retraining decision", "",
        f"Generated {decision['created_at_utc']}. Same model family as v1 (`{family}`), unchanged parameters, "
        "trained on train.csv + later.csv. Compared on the unchanged validation set.", "",
        "| Version | MAE | RMSE | R2 |", "|---|---|---|---|",
        f"| v1 | {v1_metrics['mae']:.4f} | {v1_metrics['rmse']:.4f} | {v1_metrics['r2']:.4f} |",
        f"| v2 | {v2_metrics['mae']:.4f} | {v2_metrics['rmse']:.4f} | {v2_metrics['r2']:.4f} |", "",
        f"MAE improvement (v1 - v2): {improvement:.4f} grade points. Tests passed: {tests_passed}.", "",
        f"**Decision: {decision['decision']}.** {reason}", "",
        "The recovery exercise (temporarily pointing the API at v2 and restoring v1) is a rollback drill, "
        "not approval to release a rejected candidate.", "",
    ]), encoding="utf-8")

    print(f"v1 MAE {v1_metrics['mae']:.4f} | v2 MAE {v2_metrics['mae']:.4f} | improvement {improvement:.4f}")
    print(f"Decision: {decision['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
