# Local release and recovery (story O3.3)

Machine: shared demonstration machine, Python 3.11 + Docker Desktop. Requires `artifacts/model-v1.joblib`, `model-v1.json` and `active-model.json` in `./artifacts` (copied from the model owner, not from Git).

Start:    `docker compose up --build -d`   → UI http://127.0.0.1:8501 · API http://127.0.0.1:8000/health
Logs:     `docker compose logs api` · prediction events: `logs/predictions.jsonl` (kept on the host)
Restart:  `docker compose restart api`  (the model is loaded once at API startup)
Stop:     `docker compose down`

Switch or roll back the active model:
1. `python -m src.release backup`
2. `python -m src.release activate v2`  (drill only unless the promotion rule passed)
3. `docker compose restart api`, then check `/health` shows the new version
4. Undo: `python -m src.release restore` then `docker compose restart api`; confirm `/health` shows v1 and the earlier prediction returns.

If the API is unhealthy: `docker compose logs api`; the usual causes are a missing `artifacts/` file or a wrong `active-model.json` path. Restore the backup first, then investigate.
