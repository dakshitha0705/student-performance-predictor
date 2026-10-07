"""Check that dataset partitions are complete and reproducible."""

import hashlib
import json

import pandas as pd
import pytest

from src.split import create_partitions


def sample_data():
    """Create artificial data so tests do not need the real dataset."""
    count = 100

    return pd.DataFrame({
        "source_row_id": range(1, count + 1),
        "G1": [i % 21 for i in range(count)],
        "G2": [(i + 2) % 21 for i in range(count)],
        "studytime": [(i % 4) + 1 for i in range(count)],
        "G3": [(i + 3) % 21 for i in range(count)],
    })


def test_expected_partition_names():
    parts = create_partitions(sample_data())

    assert set(parts) == {"train", "validation", "test", "later"}


def test_expected_partition_sizes():
    parts = create_partitions(sample_data())

    assert len(parts["train"]) == 60
    assert len(parts["validation"]) == 15
    assert len(parts["test"]) == 15
    assert len(parts["later"]) == 10


def test_partitions_are_disjoint():
    parts = create_partitions(sample_data())

    id_sets = {
        name: set(part["source_row_id"])
        for name, part in parts.items()
    }

    names = list(id_sets)

    for index, first in enumerate(names):
        for second in names[index + 1:]:
            assert id_sets[first].isdisjoint(id_sets[second])


def test_partitions_cover_every_cleaned_record():
    original = sample_data()
    parts = create_partitions(original)

    combined = pd.concat(parts.values(), ignore_index=True)

    assert len(combined) == len(original)
    assert combined["source_row_id"].is_unique

    expected = original.sort_values(
        "source_row_id"
    ).reset_index(drop=True)

    actual = combined.sort_values(
        "source_row_id"
    ).reset_index(drop=True)

    # Checks complete row contents, not only IDs.
    pd.testing.assert_frame_equal(actual, expected)


def test_same_seed_repeats_identically():
    original = sample_data()

    first = create_partitions(original, seed=42)
    second = create_partitions(original, seed=42)

    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_input_order_does_not_change_partitions():
    original = sample_data()
    shuffled = original.sample(
        frac=1,
        random_state=99,
    ).reset_index(drop=True)

    first = create_partitions(original, seed=42)
    second = create_partitions(shuffled, seed=42)

    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_original_dataframe_is_unchanged():
    original = sample_data()
    before = original.copy(deep=True)

    create_partitions(original)

    pd.testing.assert_frame_equal(original, before)


def test_duplicate_row_ids_are_rejected():
    original = sample_data()
    original.loc[1, "source_row_id"] = 1

    with pytest.raises(ValueError):
        create_partitions(original)


def test_manifest_matches_generated_files(tmp_path, monkeypatch):
    """Run the actual file-writing process in a temporary directory."""
    import src.split as split_module

    original = sample_data()

    raw = tmp_path / "raw.csv"
    cleaned = tmp_path / "cleaned.csv"
    validation = tmp_path / "validation.json"
    output_dir = tmp_path / "processed"
    manifest_path = tmp_path / "splits.json"

    original.drop(columns="source_row_id").to_csv(
        raw, sep=";", index=False
    )
    original.to_csv(cleaned, sep=";", index=False)

    validation.write_text(
        json.dumps({
            "valid": True,
            "cleaned_file_written": True,
            "cleaned_row_count": len(original),
        }),
        encoding="utf-8",
    )

    # Redirect the script to temporary files.
    # The real project dataset is not touched.
    monkeypatch.setattr(split_module, "RAW", raw)
    monkeypatch.setattr(split_module, "CLEANED", cleaned)
    monkeypatch.setattr(split_module, "VALIDATION_REPORT", validation)
    monkeypatch.setattr(split_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(split_module, "MANIFEST", manifest_path)

    assert split_module.main() == 0

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["seed"] == 42
    assert manifest["total_rows"] == len(original)
    assert manifest["source_sha256"] == hashlib.sha256(
        cleaned.read_bytes()
    ).hexdigest()

    all_ids = []

    for name, details in manifest["partitions"].items():
        csv_path = output_dir / f"{name}.csv"
        saved = pd.read_csv(csv_path, sep=";")

        assert details["row_count"] == len(saved)
        assert details["source_row_ids"] == (
            saved["source_row_id"].tolist()
        )
        assert details["sha256"] == hashlib.sha256(
            csv_path.read_bytes()
        ).hexdigest()

        all_ids.extend(saved["source_row_id"].tolist())

    assert len(all_ids) == len(set(all_ids)) == len(original)
    assert set(all_ids) == set(original["source_row_id"])

    # Run the complete file-writing process again.
    before_manifest = manifest_path.read_bytes()
    before_files = {
        name: (output_dir / f"{name}.csv").read_bytes()
        for name in manifest["partitions"]
    }

    assert split_module.main() == 0
    assert manifest_path.read_bytes() == before_manifest

    for name, previous_bytes in before_files.items():
        assert (
            output_dir / f"{name}.csv"
        ).read_bytes() == previous_bytes