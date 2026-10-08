# Model comparison

Generated 2026-10-08T02:13:39Z from the validation partition only. The final test partition has not been used for any decision here.

## Baselines

| Baseline | MAE | RMSE | R2 |
|---|---|---|---|
| mean_grade | 3.8069 | 4.8215 | -0.0083 |
| previous_grade | 1.2542 | 2.5578 | 0.7162 |

The stronger baseline by MAE is **previous_grade** (MAE 1.2542 grade points versus 3.8069 for mean_grade). MAE, RMSE and R2 are regression metrics measured in grade points (0-20 scale); none of them is an accuracy percentage.

## Candidate models (validation partition)

| Candidate | MAE | RMSE | R2 | MLflow run ID |
|---|---|---|---|---|
| ridge | 1.3730 | 2.4359 | 0.7426 | 3c26c11d3b8f41e89e01bc005f125aa8 |
| random_forest | 1.4807 | 2.4995 | 0.7290 | 972f2fa1c9c74b7889b3ad0a5902ecfa |

## Selection

Rule: choose the candidate with the lower validation MAE; if the absolute difference is at most 0.05 grade points, choose Ridge for simplicity.

Absolute MAE difference between candidates: 0.1077.
Selected: **ridge** (validation MAE 1.3730).

## Comparison with baselines

- Selected-model validation MAE is lower than mean_grade by 2.4339 grade points.
- Selected-model validation MAE is HIGHER than previous_grade by 0.1188 grade points.

The selected model does **not** beat the stronger baseline (previous_grade) on validation MAE, so it does not demonstrate improved predictive value. It is retained only as the educational pipeline demonstration. No improvement is claimed.

## Limitations

- Public Portuguese secondary-school data, not college-specific evidence.
- One random split (seed 42), not a chronological validation.
- Three input features only (G1, G2, studytime); no hyperparameter search.
- Final test metrics are produced once, later, by `python -m src.evaluate_final`.
