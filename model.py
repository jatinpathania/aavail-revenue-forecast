#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Model Training & Comparison
=====================================================

Model training, comparison, loading and prediction.

For every country (plus 'all' countries combined) four models are compared
on a time-ordered hold-out set (the last 20% of days):

    baseline           - DummyRegressor (predicts the training mean)
    linear             - Ridge regression
    random_forest      - RandomForestRegressor
    gradient_boosting  - GradientBoostingRegressor

The best model by RMSE is re-fit on all the data and saved. The full
comparison is saved to models/model_comparison.csv (production) or
models/test/model_comparison.csv (tests).

    python model.py            # train production models
    python model.py --test     # train test models (isolated from production)
"""

import os
import time
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cslib import fetch_ts, engineer_features, country_key, DATA_DIR
from logger import update_train_log, update_predict_log

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEST_MODEL_DIR = os.path.join(MODEL_DIR, "test")

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "best of baseline / ridge / random forest / gradient boosting"

PROD_PREFIX = "sl"
TEST_PREFIX = "test"
TEST_COUNTRIES = ("all", "united_kingdom")  # test mode trains only these (fast)


def _model_dir(test):
    return TEST_MODEL_DIR if test else MODEL_DIR


def _prefix(test):
    return TEST_PREFIX if test else PROD_PREFIX


def get_candidate_models():
    """Model name -> (pipeline, parameter grid)."""
    return {
        "baseline": (
            Pipeline([("reg", DummyRegressor(strategy="mean"))]), {}),
        "linear": (
            Pipeline([("scaler", StandardScaler()), ("reg", Ridge())]),
            {"reg__alpha": [0.1, 1.0, 10.0]}),
        "random_forest": (
            Pipeline([("scaler", StandardScaler()),
                      ("reg", RandomForestRegressor(random_state=42))]),
            {"reg__n_estimators": [50, 100], "reg__max_depth": [None, 10]}),
        "gradient_boosting": (
            Pipeline([("scaler", StandardScaler()),
                      ("reg", GradientBoostingRegressor(random_state=42))]),
            {"reg__n_estimators": [100, 200], "reg__learning_rate": [0.05, 0.1]}),
    }


def _score(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def compare_models(X, y, test_size=0.2):
    """
    Grid-search every candidate on the first (1 - test_size) of the days and
    score it on the remaining days. Returns (results DataFrame, fitted models).
    """
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    rows, fitted = [], {}
    for name, (pipe, grid) in get_candidate_models().items():
        search = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(n_splits=3),
                              scoring="neg_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        rows.append({"model": name, **_score(y_test, y_pred),
                     "best_params": str(search.best_params_)})
        fitted[name] = search.best_estimator_

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return results, fitted


def _model_train(df, tag, test=False):
    """Compare models for one time series, save the best, log the run."""
    time_start = time.time()
    X, y, dates = engineer_features(df, training=True)

    results, fitted = compare_models(X, y)
    best_name = results.loc[0, "model"]
    eval_test = {"rmse": round(float(results.loc[0, "rmse"]), 2),
                 "mae": round(float(results.loc[0, "mae"]), 2)}

    # refit the winning configuration on all the data
    best = fitted[best_name]
    best.fit(X, y)

    out_dir = _model_dir(test)
    os.makedirs(out_dir, exist_ok=True)
    version = str(MODEL_VERSION).replace(".", "_")
    path = os.path.join(out_dir, f"{_prefix(test)}-{tag}-{version}.joblib")
    joblib.dump({"model": best, "model_name": best_name, "tag": tag,
                 "version": MODEL_VERSION, "comparison": results}, path)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)
    update_train_log(tag, (str(dates[0]), str(dates[-1])), best_name, eval_test,
                     runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=test)

    results.insert(0, "country", tag)
    return results


def model_train(data_dir=DATA_DIR, test=False):
    """Train (compare + select) a model for every country and save the comparison."""
    if test:
        print("... test flag on: training test models only")

    ts_data = fetch_ts(data_dir)
    all_results = []
    for tag, df in ts_data.items():
        if test and tag not in TEST_COUNTRIES:
            continue
        print(f"... training model for {tag}")
        all_results.append(_model_train(df, tag, test=test))

    comparison = pd.concat(all_results, ignore_index=True)
    out_dir = _model_dir(test)
    comparison.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    return comparison


def model_load(data_dir=DATA_DIR, training=False, test=False):
    """
    Load saved models (production or test) and the engineered features.
    Returns (all_data, all_models) keyed by country tag.
    """
    model_dir = _model_dir(test)
    prefix = _prefix(test) + "-"

    def _saved_models():
        if not os.path.isdir(model_dir):
            return []
        return [f for f in os.listdir(model_dir)
                if f.startswith(prefix) and f.endswith(".joblib")]

    files = _saved_models()
    if not files:
        # first run on a fresh clone: train automatically so the API works
        print("... no saved models found, training them now")
        model_train(data_dir=data_dir, test=test)
        files = _saved_models()
    if not files:
        raise FileNotFoundError(
            f"no models with prefix '{prefix}' in {model_dir} - train first")

    all_models = {}
    for f in files:
        saved = joblib.load(os.path.join(model_dir, f))
        all_models[saved["tag"]] = saved

    ts_data = fetch_ts(data_dir)
    all_data = {}
    for tag, df in ts_data.items():
        if tag not in all_models:
            continue
        X, y, dates = engineer_features(df, training=training)
        all_data[tag] = {"X": X, "y": y,
                         "dates": np.datetime_as_string(dates, unit="D")}
    return all_data, all_models


def model_predict(country, year, month, day, all_models=None, all_data=None, test=False):
    """Predict revenue for the 30 days starting on the given date."""
    time_start = time.time()

    if all_models is None or all_data is None:
        all_data, all_models = model_load(training=False, test=test)

    tag = country_key(str(country))
    if tag not in all_models:
        raise ValueError(f"no model available for country '{country}'. "
                         f"Available: {sorted(all_models)}")

    target_date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    dates = all_data[tag]["dates"]
    if target_date not in dates:
        raise ValueError(f"date {target_date} not in range {dates[0]} to {dates[-1]}")

    idx = int(np.where(dates == target_date)[0][0])
    query = all_data[tag]["X"].iloc[[idx]]
    y_pred = all_models[tag]["model"].predict(query)
    y_proba = None

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)
    update_predict_log(tag, [round(float(v), 2) for v in y_pred], y_proba,
                       target_date, runtime, MODEL_VERSION, test=test)

    return {"y_pred": y_pred, "y_proba": y_proba}


def main():
    parser = argparse.ArgumentParser(description="train AAVAIL revenue models")
    parser.add_argument("--test", action="store_true", help="train isolated test models")
    args = parser.parse_args()

    comparison = model_train(test=args.test)
    print("\nmodel comparison (hold-out RMSE, lower is better):")
    print(comparison.pivot(index="country", columns="model", values="rmse").round(1))

    all_data, all_models = model_load(test=args.test)
    for tag in ("all", "united_kingdom"):
        if tag in all_models:
            date = all_data[tag]["dates"][-1]
            y, m, d = date.split("-")
            result = model_predict(tag, y, m, d, all_models, all_data, test=args.test)
            print(f"{tag}: predicted revenue for 30 days from {date} = "
                  f"{result['y_pred'][0]:,.2f}")


if __name__ == "__main__":
    main()
