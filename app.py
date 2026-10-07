"""Teacher-facing prediction form (Streamlit).

The form never loads a model: it only calls the API.

Run:  python -m streamlit run app.py
Config: API_URL environment variable (default http://127.0.0.1:8000)
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

STUDY_CHOICES = {
    "Less than 2 hours": 1,
    "2 to 5 hours": 2,
    "5 to 10 hours": 3,
    "More than 10 hours": 4,
}
LIMITATION = (
    "This educational estimate uses a public secondary-school dataset. It is not a "
    "guaranteed result and should not determine academic decisions."
)
CONNECTION_ERROR = "Prediction service is unavailable. Check that the API is running and try again."

st.set_page_config(page_title="Student Performance Predictor")
st.title("Student Performance Predictor")

if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

g1 = st.number_input("First-period grade — integer 0–20", min_value=0, max_value=20, value=12, step=1)
g2 = st.number_input("Second-period grade — integer 0–20", min_value=0, max_value=20, value=14, step=1)
study_label = st.selectbox("Weekly study time", list(STUDY_CHOICES.keys()), index=1)

if st.button("Predict final grade"):
    # Clear whatever was shown before, so a failed request can never leave a stale success on screen.
    st.session_state.result = None
    st.session_state.error = None
    payload = {"G1": int(g1), "G2": int(g2), "studytime": STUDY_CHOICES[study_label]}
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except (requests.ConnectionError, requests.Timeout):
        st.session_state.error = CONNECTION_ERROR
    else:
        if response.status_code == 200:
            body = response.json()
            st.session_state.result = {"prediction": body["prediction"], "model_version": body["model_version"]}
        elif response.status_code == 422:
            st.session_state.error = "The values entered were rejected by the service. Check that all grades are whole numbers from 0 to 20."
        else:
            st.session_state.error = CONNECTION_ERROR

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.result:
    st.success(f"Estimated final grade: {st.session_state.result['prediction']:.1f} / 20")
    st.write(f"Model version: {st.session_state.result['model_version']}")

st.caption(LIMITATION)
