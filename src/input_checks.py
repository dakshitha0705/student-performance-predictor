"""Row-level input checks used by the monitoring stage (and reusable elsewhere).

Same rules as the dataset validation in src/validate.py (G1, G2 integer 0..20,
studytime integer 1..4), but returns row-level issues for a batch of inputs and
can skip the target column. Booleans and missing values are rejected.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .common import TARGET

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
