# Setup check (story O1.3)

Second person installs from `requirements.txt` in a NEW environment and records what actually happened.

- Checked by:
- Date:
- OS and Python version (`python --version`):
- Commands run:
  1. `py -3.11 -m venv .venv-check`
  2. `.\.venv-check\Scripts\python.exe -m pip install -r requirements.txt`
  3. `.\.venv-check\Scripts\python.exe -c "import pandas, sklearn, mlflow, fastapi, streamlit, uvicorn, requests, joblib, matplotlib; print('imports ok')"`
  4. `.\.venv-check\Scripts\python.exe -m pytest -q`
- Observed result of each command:
- Problems and fixes (if any):
