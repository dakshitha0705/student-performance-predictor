"""Validate student data while preserving the original CSV."""

from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "student-mat.csv"
REPORT = ROOT / "data" / "validation.json"
CLEANED = ROOT / "data" / "processed" / "student-mat-clean.csv"

RULES = {
    "G1": (0, 20),
    "G2": (0, 20),
    "studytime": (1, 4),
    "G3": (0, 20),
}


def validate_dataframe(df):
    """Return a validation report without changing the input."""
    issues = []
    missing_counts = {}
    invalid_counts = {}

    missing_columns = [
        column for column in RULES if column not in df.columns
    ]

    for column in missing_columns:
        issues.append({
            "field": column,
            "problem": "Required column is missing",
        })

    if df.empty:
        issues.append({"problem": "Dataset contains no records"})

    # This name is reserved for the identifier added during cleaning.
    if "source_row_id" in df.columns:
        issues.append({
            "field": "source_row_id",
            "problem": "Reserved column already exists in source data",
        })

    for column, (minimum, maximum) in RULES.items():
        if column not in df.columns:
            continue

        values = df[column]
        numeric = pd.to_numeric(values, errors="coerce")

        missing = (
            values.isna()
            | values.astype(str).str.strip().eq("")
        )
        boolean = values.map(
            lambda value: isinstance(value, bool)
        )
        not_numeric = numeric.isna() & ~missing
        outside_range = (
            numeric.notna()
            & ~numeric.between(minimum, maximum)
        )
        non_integer = (
            numeric.notna()
            & numeric.between(minimum, maximum)
            & numeric.mod(1).ne(0)
        )

        missing_counts[column] = int(missing.sum())

        invalid = (
            boolean | not_numeric | outside_range | non_integer
        ) & ~missing
        invalid_counts[column] = int(invalid.sum())

        checks = [
            (missing, "Missing value"),
            (boolean & ~missing, "Boolean is not an accepted integer"),
            (not_numeric, "Value is not numeric"),
            (
                outside_range,
                f"Value must be between {minimum} and {maximum}",
            ),
            (non_integer, "Value must be a whole number"),
        ]

        for mask, message in checks:
            for position, failed in enumerate(mask):
                if failed:
                    issues.append({
                        "record": position + 1,
                        "field": column,
                        "value": str(values.iloc[position]),
                        "problem": message,
                    })

    # Compare ALL original columns, not just model input columns.
    duplicates = df.duplicated(keep="first")
    duplicate_records = [
        position + 1
        for position, duplicate in enumerate(duplicates)
        if duplicate
    ]

    return {
        "row_count": int(len(df)),
        "required_columns": list(RULES),
        "missing_columns": missing_columns,
        "missing_value_counts": missing_counts,
        "invalid_value_counts": invalid_counts,
        "exact_duplicate_count": int(duplicates.sum()),
        "duplicate_record_numbers": duplicate_records,
        "valid": len(issues) == 0,
        "issue_count": len(issues),
        "issues": issues,
    }


def clean_dataframe(df):
    """Remove exact duplicate records from a separate working copy."""
    keep = ~df.duplicated(keep="first")

    working = df.copy()
    working.insert(0, "source_row_id", range(1, len(df) + 1))

    return working.loc[keep].reset_index(drop=True)


def validate_file(dataset_path, report_path, cleaned_path):
    """Validate a CSV, save its report and write valid cleaned data."""
    dataset_path = Path(dataset_path)
    report_path = Path(report_path)
    cleaned_path = Path(cleaned_path)

    paths = [
        dataset_path.resolve(),
        report_path.resolve(),
        cleaned_path.resolve(),
    ]
    if len(set(paths)) != 3:
        raise ValueError("Source, report and cleaned paths must differ.")

    try:
        df = pd.read_csv(dataset_path, sep=";")
    except (OSError, ValueError, UnicodeError) as error:
        report = {
            "valid": False,
            "issue_count": 1,
            "issues": [{"problem": f"Cannot read dataset: {error}"}],
        }
    else:
        report = validate_dataframe(df)

    report["file_name"] = dataset_path.name
    report["cleaned_file_written"] = False

    if report["valid"]:
        cleaned = clean_dataframe(df)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(cleaned_path, sep=";", index=False)

        report["cleaned_row_count"] = int(len(cleaned))
        report["cleaned_file_written"] = True
    else:
        # Remove only a previous generated output, never the source.
        # This prevents later steps from using stale cleaned data.
        if cleaned_path.exists():
            cleaned_path.unlink()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main():
    report = validate_file(DATASET, REPORT, CLEANED)
    print(f"Report: {REPORT}")

    if not report["valid"]:
        print(f"FAIL: {report['issue_count']} validation issue(s).")
        print("Read data/validation.json for details.")
        return 1

    print(f"PASS: {report['row_count']} source records validated.")
    print(
        "Exact duplicates removed from working copy: "
        f"{report['exact_duplicate_count']}"
    )
    print(f"Cleaned records: {report['cleaned_row_count']}")
    print(f"Cleaned dataset: {CLEANED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())