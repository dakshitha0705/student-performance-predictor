"""Validate the raw mathematics dataset without changing it."""

from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "student-mat.csv"
REPORT = ROOT / "data" / "validation.json"

# Minimum and maximum accepted values, inclusive.
RULES = {
    "G1": (0, 20),
    "G2": (0, 20),
    "studytime": (1, 4),
    "G3": (0, 20),
}


def validate_dataframe(df):
    """Return a validation report for a loaded dataset."""
    issues = []

    missing_columns = [
        column for column in RULES if column not in df.columns
    ]

    for column in missing_columns:
        issues.append({
            "field": column,
            "problem": "Required column is missing",
        })

    if df.empty:
        issues.append({
            "problem": "Dataset contains no student records",
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

        checks = [
            (missing, "Missing value"),
            (not_numeric, "Value is not numeric"),
            (
                outside_range,
                f"Value must be between {minimum} and {maximum}",
            ),
            (non_integer, "Value must be a whole number"),
        ]

        for mask, message in checks:
            for position, invalid in enumerate(mask):
                if invalid:
                    issues.append({
                        # Record 1 is the first student after the header.
                        "record": position + 1,
                        "field": column,
                        "value": str(values.iloc[position]),
                        "problem": message,
                    })

    return {
        "file_name": DATASET.name,
        "row_count": int(len(df)),
        "required_columns": list(RULES),
        "missing_columns": missing_columns,
        "valid": len(issues) == 0,
        "issue_count": len(issues),
        "issues": issues,
    }


def main():
    try:
        df = pd.read_csv(DATASET, sep=";")
    except (OSError, ValueError, UnicodeError) as error:
        report = {
            "file_name": DATASET.name,
            "valid": False,
            "issue_count": 1,
            "issues": [{
                "problem": f"Unable to read dataset: {error}",
            }],
        }
    else:
        report = validate_dataframe(df)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Report saved to: {REPORT}")

    if report["valid"]:
        print(f"PASS: All {report['row_count']} records are valid.")
        return 0

    print(f"FAIL: Found {report['issue_count']} validation issue(s).")
    print("Open data/validation.json for details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())