"""Stage 3: simple baselines on the validation partition.

Run:  python -m src.baselines

  mean_grade      predicts the TRAINING-set mean of G3 for every student
  previous_grade  predicts G3 = G2 (the second-period grade) directly

Both are scored on exactly the same validation rows. Output:
  reports/baselines.csv
  reports/model-comparison.md   (baseline section; train.py extends it)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .common import REPORTS_DIR, TARGET, load_partition, regression_metrics, utc_now
from .report_text import render_model_comparison


def run_baselines(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    y_val = validation[TARGET].to_numpy(dtype=float)
    mean_pred = np.full(len(validation), float(train[TARGET].mean()))  # training targets only
    g2_pred = np.clip(validation["G2"].to_numpy(dtype=float), 0, 20)
    rows = []
    for name, pred in (("mean_grade", mean_pred), ("previous_grade", g2_pred)):
        rows.append({"baseline": name, **regression_metrics(y_val, pred),
                     "n_validation": int(len(validation))})
    return pd.DataFrame(rows)


def main() -> int:
    train = load_partition("train")
    validation = load_partition("validation")
    table = run_baselines(train, validation)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "baselines.csv", index=False)
    (REPORTS_DIR / "model-comparison.md").write_text(
        render_model_comparison(table, None, generated_at=utc_now()), encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("OK: wrote reports/baselines.csv and reports/model-comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
