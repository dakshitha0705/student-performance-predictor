"""Stage 5: demonstrate input-shift monitoring on two batches.

Run:  python -m src.monitor

Reference: TRAINING G2 mean and POPULATION standard deviation (ddof=0).
Score:     abs(batch_mean - training_mean) / max(training_std, 1)
Flag:      input_shift when score > 0.5   (demonstration threshold only)

Batches
  normal   the validation partition, unmodified
  shifted  the same rows with G2 reduced by 5 and clipped at 0. SYNTHETIC.

MAE is computed ONLY for the unmodified labelled batch. Changed inputs are never
paired with the original grades to claim anything about accuracy, and drift
alone is never described as proof of degraded accuracy.

Writes reports/monitoring.md, reports/monitoring.json, reports/g2-distribution.png
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import (  # noqa: E402
    REPORTS_DIR, ROOT, TARGET, load_active_model, load_partition, predict_clipped,
    regression_metrics, utc_now, write_json,
)
from .validate import INPUT_FIELDS, validate_frame  # noqa: E402

THRESHOLD = 0.5
SHIFT_AMOUNT = 5


def shift_score(batch_mean: float, training_mean: float, training_std: float) -> float:
    return abs(batch_mean - training_mean) / max(training_std, 1.0)


def make_shifted(batch: pd.DataFrame, amount: int = SHIFT_AMOUNT) -> pd.DataFrame:
    shifted = batch.copy()
    shifted["G2"] = (shifted["G2"] - amount).clip(lower=0)
    return shifted


def summarise_batch(name: str, batch: pd.DataFrame, ref_mean: float, ref_std: float, synthetic: bool) -> dict:
    issues = validate_frame(batch, require_target=False)
    missing = int(batch[INPUT_FIELDS].isna().sum().sum())
    mean = float(batch["G2"].mean())
    score = shift_score(mean, ref_mean, ref_std)
    return {
        "batch": name,
        "synthetic": synthetic,
        "rows": int(len(batch)),
        "missing_inputs": missing,
        "invalid_inputs": len(issues),
        "g2_mean": mean,
        "shift_score": score,
        "trigger": "input_shift" if score > THRESHOLD else "no_shift",
    }


def log_summary() -> dict | None:
    path = ROOT / "logs" / "predictions.jsonl"
    if not path.exists():
        return None
    counts: dict[str, int] = {}
    versions: dict[str, int] = {}
    total = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            counts[event.get("status", "unknown")] = counts.get(event.get("status", "unknown"), 0) + 1
            version = event.get("model_version") or "none"
            versions[version] = versions.get(version, 0) + 1
    return {"events": total, "by_status": counts, "by_model_version": versions}


def main() -> int:
    train = load_partition("train")
    validation = load_partition("validation")
    ref_mean = float(train["G2"].mean())
    ref_std = float(np.std(train["G2"].to_numpy(dtype=float), ddof=0))  # population std

    normal = validation.copy()
    shifted = make_shifted(validation)
    summaries = [
        summarise_batch("normal (validation, unmodified)", normal, ref_mean, ref_std, False),
        summarise_batch("shifted (validation G2 - 5, clipped at 0)", shifted, ref_mean, ref_std, True),
    ]

    model, version = load_active_model()
    normal_metrics = regression_metrics(normal[TARGET], predict_clipped(model, normal))

    report = {
        "created_at_utc": utc_now(),
        "reference": {"training_g2_mean": ref_mean, "training_g2_population_std": ref_std,
                      "denominator": max(ref_std, 1.0), "threshold": THRESHOLD},
        "batches": summaries,
        "active_model_version": version,
        "normal_batch_metrics": normal_metrics,
        "prediction_log_summary": log_summary(),
    }
    write_json(REPORTS_DIR / "monitoring.json", report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-0.5, 21.5, 1)
    ax.hist(train["G2"], bins=bins, alpha=0.45, density=True, label="training G2 (reference)")
    ax.hist(normal["G2"], bins=bins, histtype="step", linewidth=2, density=True, label="normal batch")
    ax.hist(shifted["G2"], bins=bins, histtype="step", linewidth=2, linestyle="--", density=True,
            label="shifted batch (SYNTHETIC)")
    ax.set_xlabel("Second-period grade (G2)")
    ax.set_ylabel("Density")
    ax.set_title("G2 distribution: reference, normal and synthetic shifted batch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "g2-distribution.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Input monitoring report", "",
        f"Generated {report['created_at_utc']}. Active model: {version}.", "",
        "## Reference", "",
        f"- Training G2 mean: {ref_mean:.4f}",
        f"- Training G2 population standard deviation: {ref_std:.4f} (denominator used: {max(ref_std, 1.0):.4f})",
        f"- Rule: flag `input_shift` when abs(batch mean - training mean) / max(std, 1) > {THRESHOLD}. "
        "This is a demonstration threshold, not a statistically calibrated production alert.", "",
        "## Batches", "",
        "| Batch | Rows | Missing inputs | Invalid inputs | G2 mean | Shift score | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(f"| {s['batch']} | {s['rows']} | {s['missing_inputs']} | {s['invalid_inputs']} | "
                     f"{s['g2_mean']:.4f} | {s['shift_score']:.4f} | {s['trigger']} |")
    lines += [
        "", "The shifted batch is **synthetic**: validation G2 reduced by "
        f"{SHIFT_AMOUNT} and clipped at 0. It was created to demonstrate the trigger, not observed from real students.", "",
        "## Model error", "",
        f"MAE on the unmodified labelled normal batch (active model {version}): {normal_metrics['mae']:.4f} grade points "
        f"(RMSE {normal_metrics['rmse']:.4f}, R2 {normal_metrics['r2']:.4f}).",
        "MAE is not reported for the shifted batch: its inputs were changed, so the original grades are no longer "
        "valid labels for it. An input-shift flag is a reason to investigate or collect new labelled data; it is not "
        "evidence that accuracy has fallen.", "",
        "## Distribution chart", "", "![G2 distribution](g2-distribution.png)", "",
    ]
    summary = report["prediction_log_summary"]
    if summary:
        lines += ["## Prediction log summary", "",
                  f"- Events: {summary['events']}",
                  f"- By status: {summary['by_status']}",
                  f"- By model version: {summary['by_model_version']}", ""]
    (REPORTS_DIR / "monitoring.md").write_text("\n".join(lines), encoding="utf-8")

    for s in summaries:
        print(f"{s['batch']:<45} score {s['shift_score']:.4f} -> {s['trigger']}")
    print("OK: wrote reports/monitoring.md, monitoring.json, g2-distribution.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
