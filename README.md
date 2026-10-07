# Student Performance Predictor

Educational MLOps project: estimate a mathematics **final grade (0–20)** from the first-period grade (`G1`), second-period grade (`G2`) and weekly study time (`studytime`, 1–4). The model is served through a FastAPI service and a Streamlit form, tracked with MLflow, packaged with Docker, and checked by CI.

**Team:** Divya, Prabhanjan, Chaitanya, Dakshitha · **Data:** UCI *Student Performance* (mathematics file only), Portuguese secondary schools — this is **not** college-specific evidence, and the estimate must not be used for academic decisions.

Jira: _paste the project URL_ · Repository: https://github.com/dakshitha0705/student-performance-predictor

---

## 1. Environment (Windows, PowerShell)

Needs Python 3.11, Git and Docker Desktop.

```powershell
git clone https://github.com/dakshitha0705/student-performance-predictor.git
cd student-performance-predictor
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked, use .\.venv\Scripts\python.exe instead of `python` below
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q                   # must pass before you do anything else
```

macOS/Linux: `python3.11 -m venv .venv && source .venv/bin/activate`, then the same pip/pytest commands.

## 2. Dataset

```powershell
python scripts/download_data.py       # downloads UCI zip, keeps only data/raw/student-mat.csv, writes data/manifest.json
```
If the download is blocked, get the zip from https://archive.ics.uci.edu/dataset/320/student+performance , put `student-mat.csv` in `data/raw/`, then run `python scripts/download_data.py --manifest-only`.
The file is **semicolon-separated** and is never modified. Raw data and model artifacts are not committed to Git; checksums are recorded instead.

## 3. Pipeline (run in this order; each command exits 0 only on success)

```powershell
python -m src.validate        # data/validation.json, data/processed/student-mat-clean.csv
python -m src.split           # data/processed/{train,validation,test,later}.csv, data/splits.json
python -m src.baselines       # reports/baselines.csv
python -m src.train           # MLflow runs, artifacts/model-v1.joblib + model-v1.json + active-model.json
python -m pytest -q
```

MLflow UI (local file store in `mlruns/`; recent MLflow versions need the opt-in variable):
```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000   # http://127.0.0.1:5000
```

## 4. Run the application locally

```powershell
# Terminal 1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# Terminal 2
python -m streamlit run app.py
```
API docs: http://127.0.0.1:8000/docs · UI: http://127.0.0.1:8501 · Health: http://127.0.0.1:8000/health

Example request: `{"G1": 12, "G2": 14, "studytime": 2}` → `{"prediction": <model output>, "scale": "0-20", "model_version": "v1"}`.
Invalid input (e.g. `G1 = -1`, booleans, unknown fields such as `G3`) returns **422**; if the model cannot be loaded, `/health` and `/predict` return **503**. The API loads the model **once at startup** — restart it after changing the active model.

## 5. Docker

```powershell
docker compose up --build -d     # UI on 8501, API on 8000
docker compose logs              # inspect
docker compose down              # stop
```
`artifacts/` is mounted read-only and `logs/` is mounted writable, so `logs/predictions.jsonl` survives restarts. Inside Docker the UI reaches the API at `http://api:8000`. The image contains no dataset, logs or model files.

## 6. Monitoring, retraining and rollback

```powershell
python -m src.monitor            # reports/monitoring.md + g2-distribution.png (normal vs synthetic shifted batch)
python -m src.retrain            # model-v2 candidate + reports/retraining-decision.md (never auto-releases)
python -m src.release backup     # copy active-model.json before any switch
python -m src.release activate v2   # rollback DRILL: restart the API and record /health + a prediction
python -m src.release restore    # put v1 back, restart the API, verify the earlier prediction returns
python -m src.evaluate_final     # run ONCE after the release choice is fixed; writes reports/final-evaluation.json
```
Promotion rule: v2 is released only if its validation MAE is **at least 0.05 lower** than v1 **and** all tests pass. Input-shift flags are demonstration signals, not proof that accuracy has fallen.

## 7. Repository layout

`api/` FastAPI service · `app.py` Streamlit form · `src/` pipeline stages · `tests/` unit tests (synthetic data only) · `scripts/` data download · `data/` manifests and (git-ignored) data · `artifacts/` model versions · `reports/` generated reports · `docs/` team records · `evidence/` dated screenshots · `logs/` prediction events (git-ignored)
