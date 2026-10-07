"""Stage 1: validate the raw mathematics CSV and write a cleaned working copy.

Run:  python -m src.validate

Exit code 0 only when every required value is valid. Rules (guide section 9):
  G1, G2   integer 0..20
  studytime integer 1..4
  G3       integer 0..20   (training only)
Missing values are rejected, never imputed. Zeros are real grades.
Only exact FULL-RECORD duplicates are removed (never rows that merely share
the three feature values).
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from .common import (
    PROCESSED_DIR, RAW_CSV, DATA_DIR, TARGET, sha256_file, utc_now, write_json,
)

RULES = {
    "G1": (0, 20),
    "G2": (0, 20),
    "studytime": (1, 4),
    "G3": (0, 20),
}
INPUT_FIELDS = ["G1", "G2", "studytime"]


def check_value(value, low: int, high: int) -> str | None:
    """Return a rejection reason, or None when the value is valid."""
    if value is None:
        return "missing"
    if isinstance(value, (bool, np.bool_)):
        return "not an integer (boolean)"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return "missing"
        if not float(value).is_integer():
            return "not an integer"
        value = int(value)
    elif isinstance(value, (int, np.integer)):
        value = int(value)
    else:
        return f"not an integer ({type(value).__name__})"
    if value < low or value > high:
        return f"out of range {low}..{high}"
    return None


def validate_frame(df: pd.DataFrame, require_target: bool = True) -> list[dict]:
    """Return a list of issues: {row, source_row_id, field, value, reason}."""
    issues: list[dict] = []
    fields = INPUT_FIELDS + ([TARGET] if require_target else [])
    for field in fields:
        if field not in df.columns:
            issues.append({"row": None, "source_row_id": None, "field": field,
                           "value": None, "reason": "column missing"})
    present = [f for f in fields if f in df.columns]
    has_id = "source_row_id" in df.columns
    for field in present:
        low, high = RULES[field]
        for index, value in df[field].items():
            reason = check_value(value, low, high)
            if reason:
                issues.append({
                    "row": int(index) if isinstance(index, (int, np.integer)) else str(index),
                    "source_row_id": int(df.at[index, "source_row_id"]) if has_id else None,
                    "field": field,
                    "value": None if pd.isna(value) else str(value),
                    "reason": reason,
                })
    return issues


def remove_full_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows identical in EVERY source column (source_row_id ignored)."""
    compare = df.drop(columns=["source_row_id"], errors="ignore")
    duplicated = compare.duplicated(keep="first")
    return df.loc[~duplicated].copy(), int(duplicated.sum())


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: {RAW_CSV} not found. Run `python scripts/download_data.py` "
              f"or place student-mat.csv in data/raw/.")
        return 1

    df = pd.read_csv(RAW_CSV, sep=";")
    if df.shape[1] < 5:
        print("ERROR: CSV parsed into too few columns. It must be read with sep=';'.")
        return 1

    # Assign the audit id BEFORE any cleaning.
    df.insert(0, "source_row_id", range(len(df)))

    required = INPUT_FIELDS + [TARGET]
    missing_counts = {f: (int(df[f].isna().sum()) if f in df.columns else None) for f in required}
    issues = validate_frame(df, require_target=True)

    report = {
        "created_at_utc": utc_now(),
        "source_file": str(RAW_CSV.relative_to(RAW_CSV.parents[2])).replace("\\", "/"),
        "dataset_sha256": sha256_file(RAW_CSV),
        "row_count_source": int(len(df)),
        "missing_value_counts": missing_counts,
        "invalid_values": issues[:500],
        "invalid_value_total": len(issues),
    }

    if issues:
        report.update({"status": "failed", "full_record_duplicates_removed": None,
                       "row_count_clean": None})
        write_json(DATA_DIR / "validation.json", report)
        print(f"FAILED: {len(issues)} invalid value(s). See data/validation.json")
        for issue in issues[:10]:
            print("  ", issue)
        return 1

    clean, removed = remove_full_duplicates(df)
    for field in required:
        clean[field] = clean[field].astype(int)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_DIR / "clean.csv", index=False)
    report.update({
        "status": "passed",
        "full_record_duplicates_removed": removed,
        "row_count_clean": int(len(clean)),
    })
    write_json(DATA_DIR / "validation.json", report)
    print(f"OK: {len(df)} source rows, {removed} full duplicate(s) removed, "
          f"{len(clean)} clean rows -> data/processed/clean.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
