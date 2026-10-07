import numpy as np
import pandas as pd
import pytest

from src.validate import check_value, remove_full_duplicates, validate_frame


def one_row(**overrides):
    row = {"source_row_id": 0, "G1": 10, "G2": 10, "studytime": 2, "G3": 10}
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.mark.parametrize("g1,g2,st,g3", [(0, 0, 1, 0), (20, 20, 4, 20), (12, 14, 2, 15)])
def test_valid_boundary_values_pass(g1, g2, st, g3):
    assert validate_frame(one_row(G1=g1, G2=g2, studytime=st, G3=g3)) == []


@pytest.mark.parametrize("field,value", [
    ("G1", -1), ("G1", 21), ("G2", -1), ("G2", 21),
    ("studytime", 0), ("studytime", 5), ("G3", -1), ("G3", 21),
])
def test_out_of_range_fails_with_row_and_field(field, value):
    issues = validate_frame(one_row(**{field: value}))
    assert len(issues) == 1
    assert issues[0]["field"] == field
    assert issues[0]["source_row_id"] == 0
    assert "out of range" in issues[0]["reason"]


@pytest.mark.parametrize("value", [12.5, "12", True, None, np.nan])
def test_wrong_type_or_missing_fails(value):
    frame = one_row()
    frame["G1"] = pd.Series([value], dtype=object)
    assert validate_frame(frame) != []


def test_missing_column_fails():
    frame = one_row().drop(columns=["studytime"])
    issues = validate_frame(frame)
    assert any(i["field"] == "studytime" and i["reason"] == "column missing" for i in issues)


def test_zero_grade_is_valid_not_missing():
    assert check_value(0, 0, 20) is None


def test_whole_number_float_is_accepted_because_nan_forces_float_dtype():
    assert check_value(12.0, 0, 20) is None


def test_prediction_validation_does_not_require_target():
    frame = one_row().drop(columns=["G3"])
    assert validate_frame(frame, require_target=False) == []


def test_only_full_record_duplicates_are_removed():
    rows = pd.DataFrame([
        {"source_row_id": 0, "G1": 10, "G2": 11, "studytime": 2, "G3": 12, "school": "GP"},
        {"source_row_id": 1, "G1": 10, "G2": 11, "studytime": 2, "G3": 12, "school": "GP"},  # exact duplicate
        {"source_row_id": 2, "G1": 10, "G2": 11, "studytime": 2, "G3": 15, "school": "GP"},  # same inputs only
        {"source_row_id": 3, "G1": 10, "G2": 11, "studytime": 2, "G3": 12, "school": "MS"},  # different other column
    ])
    cleaned, removed = remove_full_duplicates(rows)
    assert removed == 1
    assert cleaned["source_row_id"].tolist() == [0, 2, 3]
