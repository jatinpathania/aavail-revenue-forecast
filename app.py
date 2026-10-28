#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Prediction API
========================================

Flask API for the AAVAIL revenue model.

Endpoints
---------
GET  /                      health check
POST /train                 {"mode": "test"}  (omit mode for production)
POST /predict               {"query": {"country": "all", "year": "2018",
                                       "month": "01", "day": "05"},
                             "mode": "test"}
GET  /logs/<filename>       download a log file, e.g. /logs/train-test.log
GET  /monitor               drift report for the production data

Run locally:   python app.py            (http://localhost:8080)
"""

import os
import argparse

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from model import model_train, model_load, model_predict, MODEL_VERSION
from logger import LOG_DIR, TEST_LOG_DIR

app = Flask(__name__)

REQUIRED_QUERY_KEYS = ("country", "year", "month", "day")


@app.route("/", methods=["GET"])
def index():
    return jsonify({"project": "AAVAIL Revenue Forecast", "status": "ok",
                    "model_version": MODEL_VERSION})


@app.route("/train", methods=["POST"])
def train():
    payload = request.get_json(silent=True) or {}
    test = payload.get("mode") == "test"
    print("... training model", "(test mode)" if test else "")
    comparison = model_train(test=test)
    best = comparison.sort_values("rmse").groupby("country").first()["model"].to_dict()
    return jsonify({"trained": True, "test": test, "best_models": best})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "request must be JSON with a 'query' key"}), 400

    query = payload.get("query")
    if not isinstance(query, dict):
        return jsonify({"error": "'query' must be an object with keys "
                                 f"{list(REQUIRED_QUERY_KEYS)}"}), 400

    missing = [k for k in REQUIRED_QUERY_KEYS if k not in query]
    if missing:
        return jsonify({"error": f"query is missing keys: {missing}"}), 400

    test = payload.get("mode") == "test"
    try:
        all_data, all_models = model_load(training=False, test=test)
        result = model_predict(query["country"], query["year"], query["month"],
                               query["day"], all_models, all_data, test=test)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "country": query["country"],
        "target_date": f"{int(query['year']):04d}-{int(query['month']):02d}-{int(query['day']):02d}",
        "y_pred": [round(float(v), 2) for v in np.asarray(result["y_pred"])],
        "y_proba": result["y_proba"],
    })


@app.route("/logs/<filename>", methods=["GET"])
def logs(filename):
    if not filename.endswith(".log"):
        return jsonify({"error": "only .log files can be requested"}), 400
    log_dir = TEST_LOG_DIR if "test" in filename else LOG_DIR
    if not os.path.exists(os.path.join(log_dir, filename)):
        return jsonify({"error": f"log file not found: {filename}"}), 404
    return send_from_directory(log_dir, filename, as_attachment=True)


@app.route("/monitor", methods=["GET"])
def monitor():
    from monitoring import drift_report
    test = request.args.get("mode") == "test"
    report = drift_report(test=test)
    return jsonify({k: v for k, v in report.items() if not k.startswith("_")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
