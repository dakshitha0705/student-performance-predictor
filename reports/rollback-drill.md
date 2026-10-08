# R3 retraining and rollback drill

## Retraining decision

Generated on 2026-10-08 using Python 3.11.9. The v2 Ridge candidate kept the
v1 feature set and parameters, was trained on the 236 training rows plus the
40 later-batch rows, and was evaluated on the unchanged 59-row validation
partition. The test partition was not used for this decision.

| Version | Validation MAE | Validation RMSE | Validation R2 |
|---|---:|---:|---:|
| v1 | 1.3730 | 2.4359 | 0.7426 |
| v2 | 1.3987 | 2.4452 | 0.7407 |

The promotion rule requires v2 MAE to improve by at least 0.05 grade points
and the test suite to pass. Instead, v2 MAE is 0.0257 points higher than v1.
The test suite passed (76 tests), but v2 did not meet the metric threshold, so
the decision is **keep v1**. The v2 candidate was logged in MLflow as run
`30679ce1b87f4d88bb536045776aa6e1`. Training did not change
`artifacts/active-model.json`.

## Rollback drill

The active manifest was backed up, v2 was activated temporarily, and the API
was started against each active manifest to verify its loaded model version.
Both `/health` and `/predict` returned HTTP 200 for both versions. For input
`G1=12`, `G2=14`, `studytime=2`, the v2 drill returned prediction
13.8926523962. The backup was restored; the active manifest matched its
original bytes, `/health` reported v1, and the v1 prediction returned exactly
to 13.9457723883. The temporary backup was then removed. This was a recovery
drill only; the rejected v2 was not released.

The v2 metadata is in `artifacts/model-v2.json`. The local model binary
`artifacts/model-v2.joblib` is ignored by Git, consistent with repository
policy, and must be shared separately if needed for another environment.
