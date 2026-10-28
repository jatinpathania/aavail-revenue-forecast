#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Logging
=================================

Train and predict logging.

Every call to model training and to model prediction appends one row to a
CSV log file. Production logs are rotated monthly:

    logs/train-<year>-<month>.log
    logs/predict-<year>-<month>.log

Unit tests (test=True) write to a separate folder so they never touch
production logs:

    logs/test/train-test.log
    logs/test/predict-test.log
"""

import os
import csv
import time
import uuid
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
TEST_LOG_DIR = os.path.join(LOG_DIR, "test")

TRAIN_HEADER = ["unique_id", "timestamp", "tag", "period", "model_name",
                "eval_test", "runtime", "model_version", "model_version_note"]
PREDICT_HEADER = ["unique_id", "timestamp", "country", "target_date",
                  "y_pred", "y_proba", "runtime", "model_version"]


def get_log_path(kind, test=False):
    """Return the log file path for kind in {'train', 'predict'}."""
    if kind not in ("train", "predict"):
        raise ValueError(f"unknown log kind: {kind}")
    if test:
        return os.path.join(TEST_LOG_DIR, f"{kind}-test.log")
    today = date.today()
    return os.path.join(LOG_DIR, f"{kind}-{today.year}-{today.month}.log")


def _append(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def update_train_log(tag, period, model_name, eval_test, runtime,
                     model_version, model_version_note, test=False):
    """Append one row to the training log."""
    row = [uuid.uuid4(), time.time(), tag, period, model_name, eval_test,
           runtime, model_version, model_version_note]
    _append(get_log_path("train", test), TRAIN_HEADER, row)


def update_predict_log(country, y_pred, y_proba, target_date, runtime,
                       model_version, test=False):
    """Append one row to the prediction log."""
    row = [uuid.uuid4(), time.time(), country, target_date, y_pred,
           y_proba, runtime, model_version]
    _append(get_log_path("predict", test), PREDICT_HEADER, row)


if __name__ == "__main__":
    # quick manual check (writes to the TEST logs only)
    update_train_log("all", "('2017-11-29', '2019-05-31')", "random_forest",
                     {"rmse": 0.5}, "00:00:01", 0.1, "manual check", test=True)
    update_predict_log("all", [0.0], None, "2018-01-05", "00:00:01", 0.1, test=True)
    print("wrote", get_log_path("train", True), "and", get_log_path("predict", True))
