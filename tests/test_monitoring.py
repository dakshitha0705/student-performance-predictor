"""Test synthetic input shifting and monitoring calculations."""

import pandas as pd
import pytest

from src.monitor import make_shifted, shift_score, summarise_batch


def example_batch():
    return pd.DataFrame({
        "source_row_id": [1, 2, 3, 4],
        "G1": [10, 12, 14, 16],
        "G2": [0, 3, 5, 20],
        "studytime": [1, 2, 3, 4],
        "G3": [8, 10, 13, 18],
    })


def test_g2_is_reduced_and_clipped_at_zero():
    shifted = make_shifted(example_batch())

    assert shifted["G2"].tolist() == [0, 0, 0, 15]
    assert shifted["G2"].ge(0).all()


def test_original_batch_is_unchanged():
    original = example_batch()
    before = original.copy(deep=True)

    make_shifted(original)

    pd.testing.assert_frame_equal(original, before)


def test_other_columns_are_unchanged():
    original = example_batch()
    shifted = make_shifted(original)

    pd.testing.assert_frame_equal(
        original.drop(columns="G2"),
        shifted.drop(columns="G2"),
    )


def test_row_count_and_order_are_preserved():
    original = example_batch()
    shifted = make_shifted(original)

    assert len(shifted) == len(original)
    assert shifted["source_row_id"].tolist() == (
        original["source_row_id"].tolist()
    )


@pytest.mark.parametrize(
    "batch_mean,training_mean,training_std,expected",
    [
        (10.0, 10.0, 4.0, 0.0),
        (12.0, 10.0, 4.0, 0.5),
        (8.0, 10.0, 4.0, 0.5),
        (11.0, 10.0, 0.0, 1.0),
        (11.0, 10.0, 0.5, 1.0),
    ],
)
def test_shift_score_formula(
    batch_mean, training_mean, training_std, expected
):
    result = shift_score(
        batch_mean, training_mean, training_std
    )
    assert result == pytest.approx(expected)


def test_score_exactly_at_threshold_does_not_trigger():
    batch = example_batch()
    batch["G2"] = 12

    result = summarise_batch(
        "normal", batch, 10.0, 4.0, False
    )

    assert result["shift_score"] == pytest.approx(0.5)
    assert result["trigger"] == "no_shift"


def test_score_above_threshold_triggers():
    batch = example_batch()
    batch["G2"] = 13

    result = summarise_batch(
        "shifted", batch, 10.0, 4.0, True
    )

    assert result["shift_score"] == pytest.approx(0.75)
    assert result["trigger"] == "input_shift"
    assert result["synthetic"] is True


def test_summary_reports_valid_batch_correctly():
    batch = example_batch()

    result = summarise_batch(
        "normal", batch, 10.0, 4.0, False
    )

    assert result["rows"] == 4
    assert result["missing_inputs"] == 0
    assert result["invalid_inputs"] == 0
    assert result["g2_mean"] == pytest.approx(7.0)
    assert result["synthetic"] is False


def test_summary_detects_missing_input():
    batch = example_batch().astype(object)
    batch.loc[0, "G1"] = None

    result = summarise_batch(
        "normal", batch, 10.0, 4.0, False
    )

    assert result["missing_inputs"] == 1
    assert result["invalid_inputs"] >= 1


def test_summary_detects_invalid_input():
    batch = example_batch()
    batch.loc[0, "studytime"] = 5

    result = summarise_batch(
        "normal", batch, 10.0, 4.0, False
    )

    assert result["missing_inputs"] == 0
    assert result["invalid_inputs"] >= 1