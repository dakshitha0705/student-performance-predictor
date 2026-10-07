import hashlib

import pandas as pd
import pytest

from src.validate import (
    clean_dataframe,
    validate_dataframe,
    validate_file,
)


def valid_data():
    return pd.DataFrame({
        "G1": [12, 15],
        "G2": [14, 16],
        "studytime": [2, 3],
        "G3": [14, 17],
    })


def test_valid_data_passes():
    report = validate_dataframe(valid_data())
    assert report["valid"] is True
    assert report["issue_count"] == 0


def test_boundary_values_pass():
    df = pd.DataFrame({
        "G1": [0, 20],
        "G2": [0, 20],
        "studytime": [1, 4],
        "G3": [0, 20],
    })
    assert validate_dataframe(df)["valid"] is True


@pytest.mark.parametrize("field", ["G1", "G2", "studytime", "G3"])
def test_missing_required_column_fails(field):
    df = valid_data().drop(columns=[field])
    report = validate_dataframe(df)
    assert report["valid"] is False
    assert field in report["missing_columns"]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_value_fails(value):
    df = valid_data().astype(object)
    df.loc[0, "G1"] = value

    report = validate_dataframe(df)

    assert report["valid"] is False
    assert report["missing_value_counts"]["G1"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("G1", -1),
        ("G2", 21),
        ("G3", 25),
        ("studytime", 0),
        ("studytime", 5),
        ("G1", 12.5),
        ("studytime", 2.5),
        ("G2", "hello"),
        ("G1", True),
    ],
)
def test_invalid_value_fails(field, value):
    df = valid_data().astype(object)
    df.loc[0, field] = value
    assert validate_dataframe(df)["valid"] is False


def test_empty_dataset_fails():
    df = valid_data().iloc[:0]
    assert validate_dataframe(df)["valid"] is False


def test_exact_duplicate_is_removed():
    original = valid_data()
    df = pd.concat([original, original.iloc[[0]]], ignore_index=True)

    report = validate_dataframe(df)
    cleaned = clean_dataframe(df)

    assert report["valid"] is True
    assert report["exact_duplicate_count"] == 1
    assert len(cleaned) == 2
    assert cleaned["source_row_id"].tolist() == [1, 2]

    # Cleaning must not modify the input DataFrame.
    assert len(df) == 3
    assert "source_row_id" not in df.columns


def test_matching_inputs_do_not_mean_duplicate_students():
    df = pd.DataFrame({
        "G1": [12, 12],
        "G2": [14, 14],
        "studytime": [2, 2],
        "G3": [15, 15],
        "school": ["GP", "MS"],
    })

    assert validate_dataframe(df)["exact_duplicate_count"] == 0
    assert len(clean_dataframe(df)) == 2


def test_file_validation_preserves_raw_bytes(tmp_path):
    source = tmp_path / "raw.csv"
    report_path = tmp_path / "validation.json"
    cleaned_path = tmp_path / "clean.csv"

    df = valid_data()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.to_csv(source, sep=";", index=False)

    before = hashlib.sha256(source.read_bytes()).hexdigest()
    report = validate_file(source, report_path, cleaned_path)
    after = hashlib.sha256(source.read_bytes()).hexdigest()

    assert before == after
    assert report["valid"] is True
    assert report_path.exists()
    assert report["cleaned_row_count"] == 2

    cleaned = pd.read_csv(cleaned_path, sep=";")
    assert len(cleaned) == 2
    assert "source_row_id" in cleaned.columns


def test_invalid_data_removes_stale_cleaned_output(tmp_path):
    source = tmp_path / "raw.csv"
    report_path = tmp_path / "validation.json"
    cleaned_path = tmp_path / "clean.csv"

    df = valid_data()
    df.loc[0, "G1"] = -1
    df.to_csv(source, sep=";", index=False)
    cleaned_path.write_text("old generated data", encoding="utf-8")

    report = validate_file(source, report_path, cleaned_path)

    assert report["valid"] is False
    assert report["cleaned_file_written"] is False
    assert report_path.exists()
    assert not cleaned_path.exists()