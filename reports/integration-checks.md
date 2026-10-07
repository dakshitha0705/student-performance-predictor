# Integration checks (story A3)

Record what you actually saw. Compare UI and API after one-decimal rounding.

| # | Scenario | Steps | Expected | Actual outcome | Pass? |
|---|---|---|---|---|---|
| 1 | Default example, UI vs API | Submit G1=12, G2=14, study "2 to 5 hours" in the UI and `{"G1":12,"G2":14,"studytime":2}` at /docs | Same number after rounding to 1 decimal; same model version | | |
| 2 | Invalid input through the API | POST `{"G1":-1,"G2":14,"studytime":2}` | HTTP 422 | | |
| 3 | Boolean and extra field | POST `{"G1":true,...}` and `{...,"G3":15}` | HTTP 422 both | | |
| 4 | API stopped | Stop the API, press "Predict final grade" | "Prediction service is unavailable. Check that the API is running and try again." | | |
| 5 | No stale result | Make a good prediction, stop the API, predict again | The earlier number is NOT shown | | |
| 6 | Recovery | Restart the API, predict again | Normal result returns | | |
| 7 | Model missing | Start the API with a wrong `active-model.json` | `/health` and `/predict` return 503 | | |
