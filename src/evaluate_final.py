"""Stage 7: ONE final evaluation of the active model on the untouched test partition.

Run:  python -m src.evaluate_final

Run this once, after the release choice is fixed. Do not tune anything after seeing
these numbers. The script refuses to overwrite an existing result unless --force is
given, so repeated peeking is visible.
"""
from __future__ import annotations

import sys

from .common import (
    REPORTS_DIR, TARGET, load_active_model, load_partition, predict_clipped,
    regression_metrics, utc_now, write_json,
)

OUTPUT = REPORTS_DIR / "final-evaluation.json"


def main(argv: list[str]) -> int:
    if OUTPUT.exists() and "--force" not in argv:
        print(f"ERROR: {OUTPUT.name} already exists. The test set is evaluated once. "
              "Use --force only if you must repeat it, and record why.")
        return 1
    model, version = load_active_model()
    test = load_partition("test")
    metrics = regression_metrics(test[TARGET], predict_clipped(model, test))
    write_json(OUTPUT, {
        "created_at_utc": utc_now(),
        "model_version": version,
        "test_rows": int(len(test)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "note": "Single evaluation on the held-out test partition; no tuning followed.",
    })
    print(f"Model {version} on {len(test)} test rows: MAE {metrics['mae']:.4f}, "
          f"RMSE {metrics['rmse']:.4f}, R2 {metrics['r2']:.4f}")
    print("OK: wrote reports/final-evaluation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
