# Retraining decision

Generated 2026-10-08T15:48:52Z. Same model family as v1 (`ridge`), unchanged parameters, trained on train.csv + later.csv. Compared on the unchanged validation set.

| Version | MAE | RMSE | R2 |
|---|---|---|---|
| v1 | 1.3730 | 2.4359 | 0.7426 |
| v2 | 1.3987 | 2.4452 | 0.7407 |

MAE improvement (v1 - v2): -0.0257 grade points. Tests passed: True.

**Decision: keep_v1.** v2 did not meet the promotion rule (needs MAE at least 0.05 lower than v1 and passing tests); v1 is retained and v2 is recorded as a rejected candidate.

The recovery exercise (temporarily pointing the API at v2 and restoring v1) is a rollback drill, not approval to release a rejected candidate.
