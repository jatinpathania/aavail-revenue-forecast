# AAVAIL Revenue Forecast | API demo

Real requests sent to the Flask API (`app.py`) and the responses it returned.
The same calls can be made with `curl` against `python app.py` or the Docker container.

## Health check

Request:
```
GET /
```
Response (HTTP 200):
```json
{
  "model_version": 0.1,
  "project": "AAVAIL Revenue Forecast",
  "status": "ok"
}
```

## Prediction for ALL countries combined

Request:
```
POST /predict
{"query": {"country": "all", "year": "2019", "month": "07", "day": "01"}}
```
Response (HTTP 200):
```json
{
  "country": "all",
  "target_date": "2019-07-01",
  "y_pred": [
    11682.64
  ],
  "y_proba": null
}
```

## Prediction for ONE country (United Kingdom)

Request:
```
POST /predict
{"query": {"country": "united_kingdom", "year": "2019", "month": "07", "day": "01"}}
```
Response (HTTP 200):
```json
{
  "country": "united_kingdom",
  "target_date": "2019-07-01",
  "y_pred": [
    7382.39
  ],
  "y_proba": null
}
```

## Prediction for ONE country (Germany, name as written)

Request:
```
POST /predict
{"query": {"country": "Germany", "year": "2018", "month": "12", "day": "01"}}
```
Response (HTTP 200):
```json
{
  "country": "Germany",
  "target_date": "2018-12-01",
  "y_pred": [
    1126.17
  ],
  "y_proba": null
}
```

## Unknown country returns a clear error (HTTP 400)

Request:
```
POST /predict
{"query": {"country": "atlantis", "year": "2019", "month": "07", "day": "01"}}
```
Response (HTTP 400):
```json
{
  "error": "no model available for country 'atlantis'. Available: ['all', 'eire', 'france', 'germany', 'hong_kong', 'netherlands', 'norway', 'portugal', 'singapore', 'spain', 'united_kingdom']"
}
```

## Monitoring report

Request:
```
GET /monitor
```
Response (HTTP 200):
```json
{
  "feature_drift": true,
  "model_name": "random_forest",
  "n_days": 123,
  "outlier_pct": 52.03,
  "period": [
    "2019-08-01",
    "2019-12-01"
  ],
  "production_baseline_rmse": 7074.3,
  "production_rmse": 3245.85,
  "retrain_recommended": true,
  "target_drift": true,
  "train_holdout_rmse": 1030.69,
  "wasserstein_X": 1.3138,
  "wasserstein_X_threshold": 0.1675,
  "wasserstein_y": 6658.15,
  "wasserstein_y_threshold": 602.83
}
```

