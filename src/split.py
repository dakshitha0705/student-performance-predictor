"""Create reproducible partitions from the validated working dataset."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CLEANED = ROOT / "data" / "processed" / "student-mat-clean.csv"
VALIDATION_REPORT = ROOT / "data" / "validation.json"
RAW = ROOT / "data" / "raw" / "student-mat.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "splits.json"

SEED = 42
REQUIRED_COLUMNS = ["source_row_id", "G1", "G2", "studytime", "G3"]


def file_sha256(path):
    """Return a fingerprint of a file's exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_partitions(df, seed=SEED):
    """Split records without changing the input DataFrame."""
    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < 10:
        raise ValueError("At least 10 records are required.")

    ids = df["source_row_id"]

    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("source_row_id must be present and unique.")

    numeric_ids = pd.to_numeric(ids, errors="coerce")
    if (
        numeric_ids.isna().any()
        or (numeric_ids < 1).any()
        or numeric_ids.mod(1).ne(0).any()
    ):
        raise ValueError("source_row_id must contain positive integers.")

    # Sorting makes the result independent of input row ordering.
    ordered = df.copy()
    ordered["source_row_id"] = numeric_ids.astype("int64")
    ordered = ordered.sort_values("source_row_id").reset_index(drop=True)

    rng = np.random.default_rng(seed)
    positions = rng.permutation(len(ordered))

    total = len(ordered)
    later_count = int(np.ceil(total * 0.10))
    remaining_count = total - later_count

    evaluation_count = int(np.ceil(remaining_count / 3))
    test_count = int(np.ceil(evaluation_count / 2))
    validation_count = evaluation_count - test_count

    later_end = later_count
    validation_end = later_end + validation_count
    test_end = validation_end + test_count

    partitions = {
        "train": ordered.iloc[positions[test_end:]].copy(),
        "validation": ordered.iloc[
            positions[later_end:validation_end]
        ].copy(),
        "test": ordered.iloc[
            positions[validation_end:test_end]
        ].copy(),
        "later": ordered.iloc[positions[:later_end]].copy(),
    }

    # Check that no record is lost or assigned more than once.
    assigned_ids = [
        int(record_id)
        for part in partitions.values()
        for record_id in part["source_row_id"]
    ]

    if len(assigned_ids) != total:
        raise ValueError("Partition sizes do not cover all records.")

    if len(set(assigned_ids)) != total:
        raise ValueError("A record appears in multiple partitions.")

    if set(assigned_ids) != set(ordered["source_row_id"]):
        raise ValueError("Partition IDs do not match source IDs.")

    return {
        name: part.reset_index(drop=True)
        for name, part in partitions.items()
    }


def main():
    try:
        if not VALIDATION_REPORT.exists():
            raise ValueError("Run python -m src.validate first.")

        report = json.loads(
            VALIDATION_REPORT.read_text(encoding="utf-8-sig")
        )

        if (
            not report.get("valid")
            or not report.get("cleaned_file_written")
        ):
            raise ValueError("Validation must pass before splitting.")

        if not CLEANED.exists():
            raise ValueError("The cleaned dataset is missing.")

        df = pd.read_csv(CLEANED, sep=";")

        if len(df) != report.get("cleaned_row_count"):
            raise ValueError(
                "Cleaned row count differs from the validation report. "
                "Run validation again."
            )

        partitions = create_partitions(df)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        manifest = {
            "seed": SEED,
            "method": "Sorted row IDs and NumPy default_rng permutation",
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "source_file": "data/processed/student-mat-clean.csv",
            "source_sha256": file_sha256(CLEANED),
            "raw_sha256": file_sha256(RAW),
            "validation_report_sha256": file_sha256(VALIDATION_REPORT),
            "total_rows": len(df),
            "input_features": ["G1", "G2", "studytime"],
            "target": "G3",
            "identifier_excluded_from_features": "source_row_id",
            "partitions": {},
        }

        for name, part in partitions.items():
            output = OUTPUT_DIR / f"{name}.csv"

            part.to_csv(
                output,
                sep=";",
                index=False,
                encoding="utf-8",
                lineterminator="\n",
            )

            manifest["partitions"][name] = {
                "file": f"data/processed/{name}.csv",
                "row_count": len(part),
                "source_row_ids": [
                    int(value) for value in part["source_row_id"]
                ],
                "sha256": file_sha256(output),
            }

        MANIFEST.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print(f"PASS: Split {len(df)} records with seed {SEED}.")
        for name, part in partitions.items():
            print(f"{name}: {len(part)} records")

        print("No overlapping or missing record IDs.")
        print(f"Manifest: {MANIFEST}")
        return 0

    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())