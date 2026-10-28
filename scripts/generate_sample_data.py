#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Sample Data Generator
===============================================

Generate SAMPLE invoice data in the same JSON format as the official
AAVAIL capstone data (cs-train / cs-production).

Use this only if you do not have the official course data. If you do,
copy the official JSON files into data/cs-train and data/cs-production
instead and delete the sample files.

Usage:
    python scripts/generate_sample_data.py
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "cs-train")
PROD_DIR = os.path.join(BASE_DIR, "data", "cs-production")

# relative share of daily invoices per country
COUNTRIES = {
    "United Kingdom": 0.70, "EIRE": 0.05, "Germany": 0.045, "France": 0.04,
    "Norway": 0.025, "Spain": 0.02, "Hong Kong": 0.015, "Portugal": 0.015,
    "Singapore": 0.012, "Netherlands": 0.012, "Belgium": 0.01,
    "Switzerland": 0.009, "Sweden": 0.008, "Australia": 0.008,
    "Italy": 0.007, "Channel Islands": 0.006, "Denmark": 0.005,
    "Japan": 0.005, "Austria": 0.004,
}


def _generate(start, end, rng, invoice_start):
    names = list(COUNTRIES)
    probs = np.array(list(COUNTRIES.values()))
    probs = probs / probs.sum()
    rows = []
    invoice_id = invoice_start
    customers = np.arange(12000, 18000)
    for day in pd.date_range(start, end, freq="D"):
        t = (day - pd.Timestamp("2017-11-01")).days
        trend = 1 + 0.0006 * t
        season = 1 + 0.35 * np.cos(2 * np.pi * (day.dayofyear - 330) / 365.25)
        weekly = 0.45 if day.dayofweek == 5 else (0.8 if day.dayofweek == 6 else 1.0)
        n_invoices = rng.poisson(26 * trend * season * weekly)
        for _ in range(n_invoices):
            invoice_id += 1
            country = str(rng.choice(names, p=probs))
            customer = int(rng.choice(customers)) if rng.random() > 0.2 else None
            for _ in range(int(rng.integers(1, 9))):
                rows.append({
                    "country": country,
                    "customer_id": customer,
                    "day": int(day.day),
                    "invoice": str(invoice_id) if rng.random() > 0.05 else f"{invoice_id}A",
                    "month": int(day.month),
                    "price": round(float(rng.lognormal(1.0, 0.8)), 2),
                    "stream_id": str(int(rng.integers(20000, 23999))),
                    "times_viewed": int(rng.integers(1, 25)),
                    "year": int(day.year),
                })
    return pd.DataFrame(rows), invoice_id


def _write_monthly(df, out_dir, legacy_names_months=()):
    os.makedirs(out_dir, exist_ok=True)
    for (year, month), grp in df.groupby(["year", "month"]):
        records = grp.to_dict(orient="records")
        for r in records:
            if r["customer_id"] is not None and r["customer_id"] == r["customer_id"]:
                r["customer_id"] = int(r["customer_id"])
            else:
                r["customer_id"] = None
        # a few months use the legacy column names found in the official data
        if (year, month) in legacy_names_months:
            for r in records:
                r["StreamID"] = r.pop("stream_id")
                r["TimesViewed"] = r.pop("times_viewed")
                r["total_price"] = r.pop("price")
        path = os.path.join(out_dir, f"invoices-{year}-{month:02d}.json")
        with open(path, "w") as f:
            json.dump(records, f)
        print(f"wrote {path} ({len(records)} rows)")


def main(seed=42):
    rng = np.random.default_rng(seed)
    train, last_id = _generate("2017-11-28", "2019-07-31", rng, 489000)
    _write_monthly(train, TRAIN_DIR, legacy_names_months={(2019, 1), (2019, 2)})
    prod, _ = _generate("2019-08-01", "2019-12-31", rng, last_id)
    _write_monthly(prod, PROD_DIR)


if __name__ == "__main__":
    main()
