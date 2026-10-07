"""Builds reports/model-comparison.md from measured numbers only.

Every sentence is generated from the metrics passed in, so the report cannot
claim an improvement that the numbers do not show.
"""
from __future__ import annotations

import pandas as pd


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_model_comparison(baselines: pd.DataFrame, candidates: dict | None,
                            generated_at: str) -> str:
    lines = [
        "# Model comparison",
        "",
        f"Generated {generated_at} from the validation partition only. "
        "The final test partition has not been used for any decision here.",
        "",
        "## Baselines",
        "",
        "| Baseline | MAE | RMSE | R2 |",
        "|---|---|---|---|",
    ]
    for _, row in baselines.iterrows():
        lines.append(f"| {row['baseline']} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} | {_fmt(row['r2'])} |")

    best = baselines.sort_values("mae").iloc[0]
    other = baselines.sort_values("mae").iloc[1]
    lines += [
        "",
        f"The stronger baseline by MAE is **{best['baseline']}** "
        f"(MAE {_fmt(best['mae'])} grade points versus {_fmt(other['mae'])} for {other['baseline']}). "
        "MAE, RMSE and R2 are regression metrics measured in grade points (0-20 scale); "
        "none of them is an accuracy percentage.",
    ]

    if candidates:
        lines += ["", "## Candidate models (validation partition)", "",
                  "| Candidate | MAE | RMSE | R2 | MLflow run ID |", "|---|---|---|---|---|"]
        for name, info in candidates["results"].items():
            m = info["metrics"]
            lines.append(f"| {name} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {_fmt(m['r2'])} | {info['run_id']} |")
        chosen = candidates["chosen"]
        chosen_mae = candidates["results"][chosen]["metrics"]["mae"]
        lines += [
            "",
            "## Selection",
            "",
            "Rule: choose the candidate with the lower validation MAE; if the absolute "
            "difference is at most 0.05 grade points, choose Ridge for simplicity.",
            "",
            f"Absolute MAE difference between candidates: {_fmt(candidates['mae_difference'])}.",
            f"Selected: **{chosen}** (validation MAE {_fmt(chosen_mae)}).",
            "",
            "## Comparison with baselines",
            "",
        ]
        for _, row in baselines.iterrows():
            delta = row["mae"] - chosen_mae
            if delta > 0:
                verdict = f"lower than {row['baseline']} by {_fmt(delta)} grade points"
            elif delta < 0:
                verdict = f"HIGHER than {row['baseline']} by {_fmt(-delta)} grade points"
            else:
                verdict = f"equal to {row['baseline']}"
            lines.append(f"- Selected-model validation MAE is {verdict}.")
        if chosen_mae >= best["mae"]:
            lines += [
                "",
                f"The selected model does **not** beat the stronger baseline ({best['baseline']}) on "
                "validation MAE, so it does not demonstrate improved predictive value. It is retained "
                "only as the educational pipeline demonstration. No improvement is claimed.",
            ]
        else:
            lines += [
                "",
                f"The selected model has lower validation MAE than the stronger baseline "
                f"({best['baseline']}). This is a single random split of a small dataset, so the "
                "size of the difference should be read cautiously.",
            ]
        lines += [
            "",
            "## Limitations",
            "",
            "- Public Portuguese secondary-school data, not college-specific evidence.",
            "- One random split (seed 42), not a chronological validation.",
            "- Three input features only (G1, G2, studytime); no hyperparameter search.",
            "- Final test metrics are produced once, later, by `python -m src.evaluate_final`.",
        ]
    lines.append("")
    return "\n".join(lines)
