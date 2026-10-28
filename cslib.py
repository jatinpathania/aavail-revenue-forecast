#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Data Ingestion
========================================

Data ingestion and time-series engineering for the AAVAIL revenue project.

All ingestion is wrapped in functions so it can be automated (called from
model training, the API, the monitoring script, the notebooks and the
Docker build). It can also be run on its own:

    python cslib.py                 # ingest data/cs-train and cache time series
    python cslib.py --data-dir data/cs-production --clean
"""

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cs-train")
PROD_DATA_DIR = os.path.join(BASE_DIR, "data", "cs-production")

COLUMNS = ["country", "customer_id", "day", "invoice", "month",
           "price", "stream_id", "times_viewed", "year"]

# the official data uses inconsistent column names across months
COLUMN_RENAMES = {"StreamID": "stream_id", "TimesViewed": "times_viewed",
                  "total_price": "price"}

TOP_N_COUNTRIES = 10


def fetch_data(data_dir):
    """Load every JSON invoice file in data_dir into one clean DataFrame."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    if not files:
        raise FileNotFoundError(f"no .json files found in {data_dir}")

    frames = []
    for name in files:
        df = pd.read_json(os.path.join(data_dir, name))
        df = df.rename(columns=COLUMN_RENAMES)
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        frames.append(df[COLUMNS])

    df = pd.concat(frames, ignore_index=True)

    # build a proper date column and drop rows with impossible dates
    df["invoice_date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.dropna(subset=["invoice_date"])

    # invoice ids sometimes carry letters (e.g. '489434A'); keep the digits only
    df["invoice"] = df["invoice"].astype(str).map(lambda s: re.sub(r"\D+", "", s))
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce").fillna(0)

    df = df.sort_values("invoice_date").reset_index(drop=True)
    return df


def convert_to_ts(df_orig, country=None):
    """Aggregate transactions into a daily time series (optionally for one country)."""
    if country is not None:
        if country not in df_orig["country"].unique():
            raise ValueError(f"country not found: {country}")
        df = df_orig[df_orig["country"] == country]
    else:
        df = df_orig

    days = pd.date_range(df_orig["invoice_date"].min(),
                         df_orig["invoice_date"].max(), freq="D")
    grouped = df.groupby("invoice_date")
    ts = pd.DataFrame({
        "purchases": grouped.size(),
        "unique_invoices": grouped["invoice"].nunique(),
        "unique_streams": grouped["stream_id"].nunique(),
        "total_views": grouped["times_viewed"].sum(),
        "revenue": grouped["price"].sum(),
    }).reindex(days, fill_value=0)
    ts.index.name = "date"
    return ts.reset_index()


def country_key(country):
    """'United Kingdom' -> 'united_kingdom'."""
    return re.sub(r"\s+", "_", country.strip().lower())


def fetch_ts(data_dir, clean=False):
    """
    Return {name: daily time series} for all countries combined ('all')
    and the top 10 countries by revenue. Results are cached as CSV in
    <data_dir>/ts-data so later runs are fast; use clean=True to rebuild.
    """
    ts_dir = os.path.join(data_dir, "ts-data")

    if not clean and os.path.isdir(ts_dir) and os.listdir(ts_dir):
        return {
            os.path.splitext(f)[0].replace("ts-", ""):
                pd.read_csv(os.path.join(ts_dir, f), parse_dates=["date"])
            for f in sorted(os.listdir(ts_dir)) if f.endswith(".csv")
        }

    print(f"... processing data for {data_dir}")
    df = fetch_data(data_dir)

    top = (df.groupby("country")["price"].sum()
             .sort_values(ascending=False).head(TOP_N_COUNTRIES).index)

    os.makedirs(ts_dir, exist_ok=True)
    for f in os.listdir(ts_dir):
        os.remove(os.path.join(ts_dir, f))

    dfs = {"all": convert_to_ts(df)}
    for country in top:
        dfs[country_key(country)] = convert_to_ts(df, country=country)

    for key, ts in dfs.items():
        ts.to_csv(os.path.join(ts_dir, f"ts-{key}.csv"), index=False)
    return dfs


def engineer_features(df, training=True):
    """
    Turn a daily time series into a supervised learning problem.

    Features (for each day): revenue in the previous 7, 14, 28 and 70 days,
    revenue in the same 30-day window one year earlier, and the average
    number of invoices and views over the previous 30 days.

    Target: total revenue over the NEXT 30 days.

    When training=True the last 30 days are dropped, because their target
    would be incomplete.
    """
    dates = df["date"].values.astype("datetime64[D]")
    revenue = df["revenue"].values.astype(float)
    invoices = df["unique_invoices"].values.astype(float)
    views = df["total_views"].values.astype(float)

    eng = defaultdict(list)
    y = np.zeros(dates.size)
    for i, day in enumerate(dates):
        for num in (7, 14, 28, 70):
            mask = (dates >= day - np.timedelta64(num, "D")) & (dates < day)
            eng[f"previous_{num}"].append(revenue[mask].sum())

        mask = (dates >= day) & (dates < day + np.timedelta64(30, "D"))
        y[i] = revenue[mask].sum()

        start = day - np.timedelta64(365, "D")
        mask = (dates >= start) & (dates < start + np.timedelta64(30, "D"))
        eng["previous_year"].append(revenue[mask].sum())

        mask = (dates >= day - np.timedelta64(30, "D")) & (dates < day)
        eng["recent_invoices"].append(invoices[mask].mean() if mask.any() else 0.0)
        eng["recent_views"].append(views[mask].mean() if mask.any() else 0.0)

    X = pd.DataFrame(eng).fillna(0)

    if training:
        keep = dates <= dates.max() - np.timedelta64(30, "D")
        X, y, dates = X[keep], y[keep], dates[keep]

    return X.reset_index(drop=True), y, dates


def main():
    parser = argparse.ArgumentParser(description="ingest AAVAIL invoice data")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--clean", action="store_true", help="rebuild the cached time series")
    args = parser.parse_args()

    ts = fetch_ts(args.data_dir, clean=args.clean)
    for key, df in ts.items():
        print(f"{key:>20}: {df.shape[0]} days, revenue = {df['revenue'].sum():,.0f}")


if __name__ == "__main__":
    main()
