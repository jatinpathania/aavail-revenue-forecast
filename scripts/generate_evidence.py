#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Evidence Generator
============================================

Runs the project and saves the real output to reports/evidence/ so reviewers
can see that everything works without running anything themselves:

    test_results.txt       output of `python run-tests.py -v`
    api_demo.md            real requests and responses from the API
    model_comparison.md    hold-out RMSE of every model for every country
    monitoring_report.txt  output of `python monitoring.py`

Usage (from the project root, after `python model.py`):
    python scripts/generate_evidence.py
"""

import os
import sys
import json
import subprocess

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(BASE, "reports", "evidence")
sys.path.insert(0, BASE)


def run(cmd):
    result = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


def api_demo():
    from app import app
    client = app.test_client()

    calls = [
        ("GET", "/", None, "Health check"),
        ("POST", "/predict",
         {"query": {"country": "all", "year": "2019", "month": "07", "day": "01"}},
         "Prediction for ALL countries combined"),
        ("POST", "/predict",
         {"query": {"country": "united_kingdom", "year": "2019", "month": "07", "day": "01"}},
         "Prediction for ONE country (United Kingdom)"),
        ("POST", "/predict",
         {"query": {"country": "Germany", "year": "2018", "month": "12", "day": "01"}},
         "Prediction for ONE country (Germany, name as written)"),
        ("POST", "/predict",
         {"query": {"country": "atlantis", "year": "2019", "month": "07", "day": "01"}},
         "Unknown country returns a clear error (HTTP 400)"),
        ("GET", "/monitor", None, "Monitoring report"),
    ]

    lines = ["# AAVAIL Revenue Forecast | API demo",
             "",
             "Real requests sent to the Flask API (`app.py`) and the responses it returned.",
             "The same calls can be made with `curl` against `python app.py` or the Docker container.",
             ""]
    for method, url, body, title in calls:
        r = client.get(url) if method == "GET" else client.post(url, json=body)
        lines += [f"## {title}", "", "Request:", "```",
                  f"{method} {url}" + (f"\n{json.dumps(body)}" if body else ""), "```",
                  f"Response (HTTP {r.status_code}):", "```json",
                  json.dumps(r.get_json(), indent=2), "```", ""]
    return "\n".join(lines)


def model_comparison():
    import pandas as pd
    comp = pd.read_csv(os.path.join(BASE, "models", "model_comparison.csv"))
    pivot = comp.pivot(index="country", columns="model", values="rmse").round(1)
    pivot = pivot[["baseline", "linear", "random_forest", "gradient_boosting"]]
    best = comp.sort_values("rmse").groupby("country").first()["model"]
    pivot["selected_model"] = best

    header = "| country | " + " | ".join(pivot.columns) + " |"
    sep = "|" + "---|" * (len(pivot.columns) + 1)
    rows = [f"| {c} | " + " | ".join(str(v) for v in r) + " |" for c, r in pivot.iterrows()]
    return "\n".join([
        "# AAVAIL Revenue Forecast | Model comparison",
        "",
        "Four models were compared for every country on a time-ordered hold-out set",
        "(last 20% of days). Values are RMSE of 30-day revenue (lower is better).",
        "The model with the lowest RMSE is saved and served by the API.",
        "",
        header, sep, *rows, "",
        "Charts: `reports/figures/model_vs_baseline_rmse.png`, "
        "`model_vs_baseline_holdout.png`, `model_vs_baseline_by_country.png`.",
        ""])


def main():
    os.makedirs(OUT, exist_ok=True)
    py = sys.executable

    files = {
        "test_results.txt": run([py, "run-tests.py", "-v"]),
        "monitoring_report.txt": run([py, "monitoring.py"]),
        "api_demo.md": api_demo(),
        "model_comparison.md": model_comparison(),
    }
    for name, text in files.items():
        with open(os.path.join(OUT, name), "w") as f:
            f.write(text + "\n")
        print("wrote", os.path.join("reports", "evidence", name))


if __name__ == "__main__":
    main()
