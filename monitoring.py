#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Performance Monitoring
================================================

Post-production monitoring.

Three checks are provided:

1. Data drift  - an IsolationForest fitted on the training features flags the
   share of production days that look like outliers, and the Wasserstein
   distance between training and production distributions (features and
   target) is compared to a threshold learned by bootstrapping the
   training data.

2. Model performance - the saved 'all' model is scored on the production
   period (data/cs-production) and its RMSE is compared to both the
   hold-out RMSE from training and to the baseline model.

3. Log summary - counts and run times from the train / predict logs.

    python monitoring.py            # prints the report, saves a figure
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from cslib import fetch_data, convert_to_ts, engineer_features, DATA_DIR, PROD_DATA_DIR
from model import model_load
from logger import get_log_path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def get_monitoring_tools(X, y, n_bootstrap=200, seed=42):
    """Fit an outlier detector and learn drift thresholds from training data."""
    rng = np.random.default_rng(seed)
    detector = Pipeline([("scaler", StandardScaler()),
                         ("iforest", IsolationForest(contamination=0.05, random_state=seed))])
    detector.fit(X)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    half = len(X) // 2
    x_dist, y_dist = [], []
    for _ in range(n_bootstrap):
        a = rng.choice(len(X), half, replace=True)
        b = rng.choice(len(X), half, replace=True)
        x_dist.append(np.mean([wasserstein_distance(Xs[a, j], Xs[b, j])
                               for j in range(Xs.shape[1])]))
        y_dist.append(wasserstein_distance(y[a], y[b]))

    return {
        "detector": detector, "scaler": scaler,
        "wasserstein_X_threshold": float(np.percentile(x_dist, 95)),
        "wasserstein_y_threshold": float(np.percentile(y_dist, 95)),
    }


def _production_features(train_dir, prod_dir):
    """Engineer features for the production period, using training data as history."""
    train_df = fetch_data(train_dir)
    prod_df = fetch_data(prod_dir)
    combined = pd.concat([train_df, prod_df], ignore_index=True)
    ts = convert_to_ts(combined)
    X, y, dates = engineer_features(ts, training=True)
    mask = dates >= prod_df["invoice_date"].min().to_datetime64().astype("datetime64[D]")
    return X[mask].reset_index(drop=True), y[mask], dates[mask]


def drift_report(train_dir=DATA_DIR, prod_dir=PROD_DATA_DIR, test=False):
    """Compare production data with training data and score the model on it."""
    all_data, all_models = model_load(data_dir=train_dir, training=True, test=test)
    X_train, y_train = all_data["all"]["X"], all_data["all"]["y"]
    tools = get_monitoring_tools(X_train, y_train)

    X_prod, y_prod, dates = _production_features(train_dir, prod_dir)

    outlier_pct = float((tools["detector"].predict(X_prod) == -1).mean() * 100)
    Xs_train = tools["scaler"].transform(X_train)
    Xs_prod = tools["scaler"].transform(X_prod)
    w_X = float(np.mean([wasserstein_distance(Xs_train[:, j], Xs_prod[:, j])
                         for j in range(Xs_train.shape[1])]))
    w_y = float(wasserstein_distance(y_train, y_prod))

    saved = all_models["all"]
    y_pred = saved["model"].predict(X_prod)
    prod_rmse = _rmse(y_prod, y_pred)
    baseline_rmse = _rmse(y_prod, np.full_like(y_prod, y_train.mean()))
    comparison = saved["comparison"]
    train_rmse = float(comparison.loc[comparison["model"] == saved["model_name"], "rmse"].iloc[0])

    return {
        "period": [str(dates[0]), str(dates[-1])],
        "n_days": int(len(dates)),
        "outlier_pct": round(outlier_pct, 2),
        "wasserstein_X": round(w_X, 4),
        "wasserstein_X_threshold": round(tools["wasserstein_X_threshold"], 4),
        "wasserstein_y": round(w_y, 2),
        "wasserstein_y_threshold": round(tools["wasserstein_y_threshold"], 2),
        "feature_drift": bool(w_X > tools["wasserstein_X_threshold"]),
        "target_drift": bool(w_y > tools["wasserstein_y_threshold"]),
        "model_name": saved["model_name"],
        "train_holdout_rmse": round(train_rmse, 2),
        "production_rmse": round(prod_rmse, 2),
        "production_baseline_rmse": round(baseline_rmse, 2),
        # retrain when production error is 50% worse than at training time
        "retrain_recommended": bool(prod_rmse > 1.5 * train_rmse),
        "_series": {"dates": [str(d) for d in dates],
                    "y_true": [float(v) for v in y_prod],
                    "y_pred": [float(v) for v in y_pred],
                    "y_baseline": float(y_train.mean())},
    }


def summarize_logs(test=False):
    """Simple counts and runtimes from the train / predict logs."""
    summary = {}
    for kind in ("train", "predict"):
        path = get_log_path(kind, test=test)
        if not os.path.exists(path):
            summary[kind] = {"entries": 0}
            continue
        df = pd.read_csv(path)
        secs = df["runtime"].astype(str).str.split(":").apply(
            lambda p: int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2]))
        summary[kind] = {"log": os.path.basename(path), "entries": int(len(df)),
                         "mean_runtime_s": round(float(secs.mean()), 2)}
        if kind == "train" and "eval_test" in df:
            rmses = df["eval_test"].astype(str).str.extract(r"'rmse':\s*([\d.]+)")[0]
            summary[kind]["mean_rmse"] = round(float(rmses.astype(float).mean()), 2)
    return summary


def plot_production_performance(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = report["_series"]
    dates = pd.to_datetime(s["dates"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, s["y_true"], label="actual (next 30 days)", color="black", lw=2)
    ax.plot(dates, s["y_pred"], label=f"model: {report['model_name']}", color="tab:blue")
    ax.axhline(s["y_baseline"], label="baseline (training mean)", color="tab:red", ls="--")
    ax.set_title(f"Production performance - RMSE model {report['production_rmse']:,.0f} "
                 f"vs baseline {report['production_baseline_rmse']:,.0f}")
    ax.set_ylabel("revenue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="monitor model performance and drift")
    parser.add_argument("--test", action="store_true", help="use the test models / logs")
    args = parser.parse_args()

    report = drift_report(test=args.test)
    print("\n=== drift and performance report ===")
    for k, v in report.items():
        if not k.startswith("_"):
            print(f"{k:>28}: {v}")

    print("\n=== log summary ===")
    for k, v in summarize_logs(test=args.test).items():
        print(f"{k:>28}: {v}")

    if not args.test:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "production_performance.png")
        plot_production_performance(report, path)
        print(f"\nsaved {os.path.relpath(path, BASE_DIR)}")


if __name__ == "__main__":
    main()
