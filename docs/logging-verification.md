# R1 Prediction Logging Verification

Owner: Divya
Reviewer: Chaitanya

## Implementation

The API implements logging for POST /predict.

Each event records:
- request_id
- timestamp_utc
- status
- latency_ms
- model_version, or null when unavailable

Successful events additionally record:
- validated inputs
- prediction

Validation-error and service-error events omit inputs and predictions.
Raw request bodies are not logged.

Output location: logs/predictions.jsonl
Deployment assumption: one API worker.

## Automated verification

Command:
python -m pytest tests/test_logging.py -q

Observed result:
10 passed, 1 warning in 0.96s

Warning:
Starlette TestClient reported deprecated httpx usage.

The tests use a substitute model and temporary log files.
They do not establish that the real trained model loads successfully.

Verified scenarios:
- Successful prediction logging.
- Unique request IDs.
- UTC timestamps and nonnegative latency.
- Rejected inputs without raw payload logging.
- Unavailable-model service errors.
- Prediction-failure service errors.

## Pending verification

- Real model loads and GET /health returns 200.
- Valid prediction returns 200 and creates a success log.
- Invalid prediction returns 422 and creates a validation-error log.
- Logs persist after Docker restart.

## Review

Chaitanya's review: pending.
Real-model checks: blocked by the M3 model package.
Docker persistence check: pending deployment availability.