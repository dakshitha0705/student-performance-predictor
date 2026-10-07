import joblib
import numpy as np
import pandas as pd

from src.common import FEATURES, load_active_model, predict_clipped, regression_metrics
from src.retrain import should_promote
from src.train import build_candidates, choose_candidate

from tests.conftest import synthetic_frame


def test_reloaded_pipeline_matches_original(tmp_path, tiny_pipeline):
    path = tmp_path / "model.joblib"
    joblib.dump(tiny_pipeline, path)
    reloaded = joblib.load(path)
    sample = synthetic_frame(50, seed=3)
    original = predict_clipped(tiny_pipeline, sample)
    again = predict_clipped(reloaded, sample)
    assert np.max(np.abs(original - again)) < 1e-6


def test_active_manifest_loads_model_and_version(active_manifest):
    model, version = load_active_model(active_manifest)
    assert version == "v1"
    assert hasattr(model, "predict")


def test_predictions_are_clipped_to_grade_range():
    class Extreme:
        def predict(self, X):
            return np.array([-5.0, 25.0, 10.0])

    frame = pd.DataFrame({"G1": [0, 20, 10], "G2": [0, 20, 10], "studytime": [1, 4, 2]})
    out = predict_clipped(Extreme(), frame)
    assert out.min() >= 0 and out.max() <= 20
    assert out[2] == 10.0  # in-range values are not rounded or altered


def test_feature_order_is_enforced():
    class Capture:
        columns = None

        def predict(self, X):
            Capture.columns = list(X.columns)
            return np.zeros(len(X))

    shuffled = pd.DataFrame({"studytime": [2], "G2": [14], "G1": [12]})
    predict_clipped(Capture(), shuffled)
    assert Capture.columns == FEATURES


def test_metrics_perfect_prediction():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["mae"] == 0 and m["rmse"] == 0 and m["r2"] == 1


def test_candidates_use_the_agreed_settings():
    candidates = build_candidates()
    ridge = candidates["ridge"].named_steps["model"]
    forest = candidates["random_forest"].named_steps["model"]
    assert ridge.alpha == 1.0
    assert (forest.n_estimators, forest.max_depth, forest.min_samples_leaf) == (200, 5, 3)
    assert forest.random_state == 42 and forest.n_jobs == 1


def _result(mae):
    return {"metrics": {"mae": mae}}


def test_tie_within_005_prefers_ridge():
    chosen, diff = choose_candidate({"ridge": _result(1.00), "random_forest": _result(0.96)})
    assert chosen == "ridge" and diff <= 0.05


def test_clear_winner_is_chosen():
    chosen, _ = choose_candidate({"ridge": _result(1.20), "random_forest": _result(1.00)})
    assert chosen == "random_forest"
    chosen, _ = choose_candidate({"ridge": _result(0.90), "random_forest": _result(1.00)})
    assert chosen == "ridge"


def test_promotion_rule():
    assert should_promote(1.00, 0.90, tests_passed=True)
    assert not should_promote(1.00, 0.96, tests_passed=True)    # improvement < 0.05
    assert not should_promote(1.00, 0.90, tests_passed=False)   # tests failed
