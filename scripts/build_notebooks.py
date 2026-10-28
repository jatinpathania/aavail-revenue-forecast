#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Notebook Builder
==========================================

Build (and optionally execute) the project notebooks.

    python scripts/build_notebooks.py           # write notebooks
    python scripts/build_notebooks.py --execute # write + run them (embeds plots)
"""
import os
import sys
import nbformat as nbf

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_DIR = os.path.join(BASE, "notebooks")

SETUP = """import os, sys
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
FIG_DIR = os.path.join('..', 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
%matplotlib inline"""


def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())


eda = [
    md("""
# AAVAIL Revenue Forecast | 01 - Exploratory Data Analysis (EDA)

**Business question:** AAVAIL management wants to predict the revenue for the
following month, for any country or for all countries combined, so staff can
plan resources.

**Data:** monthly JSON invoice files in `data/cs-train`. Ingestion is automated
by `cslib.fetch_data()` (raw transactions) and `cslib.fetch_ts()` (daily time
series per country). This notebook uses those same functions.
"""),
    code(SETUP),
    md("## 1. Ingest the data"),
    code("""from cslib import fetch_data, fetch_ts, DATA_DIR
df = fetch_data(DATA_DIR)
print(df.shape)
print('date range:', df.invoice_date.min().date(), 'to', df.invoice_date.max().date())
print('countries:', df.country.nunique())
df.head()"""),
    md("## 2. Data quality: missing values and summary statistics"),
    code("""missing = df.isnull().mean().mul(100).round(2).rename('% missing')
display(missing.to_frame())
df[['price', 'times_viewed']].describe()"""),
    md("## 3. Revenue over time (monthly)"),
    code("""monthly = df.set_index('invoice_date')['price'].resample('MS').sum()
days_per_month = df.groupby(df.invoice_date.dt.to_period('M')).invoice_date.nunique()
monthly = monthly[days_per_month.values >= 20]  # drop partial months at the edges
fig, ax = plt.subplots(figsize=(11, 4))
monthly.plot(ax=ax, marker='o')
ax.set_title('Total monthly revenue (all countries)')
ax.set_ylabel('revenue'); ax.set_xlabel('')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_monthly_revenue.png'), dpi=120)
plt.show()"""),
    md("## 4. Top 10 countries by revenue"),
    code("""by_country = df.groupby('country')['price'].sum().sort_values(ascending=False)
top10 = by_country.head(10)
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(x=top10.values, y=top10.index, ax=ax, color='tab:blue')
ax.set_title('Top 10 countries by total revenue'); ax.set_xlabel('revenue'); ax.set_ylabel('')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_top10_countries.png'), dpi=120)
plt.show()
print(f"Top 10 countries = {top10.sum() / by_country.sum():.1%} of revenue")"""),
    md("## 5. Daily revenue and 30-day rolling average for the top countries"),
    code("""ts = fetch_ts(DATA_DIR)
keys = ['all'] + [k for k in ts if k != 'all'][:5]
fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.3 * len(keys)), sharex=True)
for ax, key in zip(axes, keys):
    s = ts[key].set_index('date')['revenue']
    ax.plot(s.index, s.values, alpha=0.35, lw=0.8)
    ax.plot(s.index, s.rolling(30).mean(), lw=2)
    ax.set_title(key, loc='left', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_daily_revenue_by_country.png'), dpi=120)
plt.show()"""),
    md("## 6. Weekly pattern"),
    code("""daily = ts['all'].copy()
daily['weekday'] = daily['date'].dt.day_name()
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
fig, ax = plt.subplots(figsize=(9, 4))
sns.boxplot(data=daily, x='weekday', y='revenue', order=order, ax=ax, color='tab:blue')
ax.set_title('Daily revenue by day of week'); ax.set_xlabel('')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_weekday.png'), dpi=120)
plt.show()"""),
    md("## 7. Relationships between daily features"),
    code("""corr = ts['all'][['purchases','unique_invoices','unique_streams','total_views','revenue']].corr()
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax)
ax.set_title('Correlation of daily features')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_correlation.png'), dpi=120)
plt.show()"""),
    md("## 8. Distribution of daily revenue"),
    code("""fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(ts['all']['revenue'], bins=40, kde=True, ax=ax)
ax.set_title('Distribution of daily revenue (all countries)')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_revenue_distribution.png'), dpi=120)
plt.show()"""),
    md("""
## 9. Findings

* Revenue is concentrated in a few countries (the United Kingdom dominates), so
  the API serves models for the **top 10 countries plus all countries combined**.
* Revenue has a clear **seasonal pattern** (peaks toward the end of the year)
  and a **weekly pattern** (lower at weekends), so features based on recent
  revenue windows and on the same period last year are useful.
* The number of invoices, streams and views move closely with revenue, so
  recent activity is included as a feature.
* Daily revenue is noisy; predicting the **sum over the next 30 days** smooths
  this out and matches the business question (next month's revenue).

**Hypotheses to test in modelling:** a non-linear model using recent revenue
windows will beat a naive baseline (predicting the average), and the advantage
will be largest for the countries with the most data.
"""),
]

model_nb = [
    md("""
# AAVAIL Revenue Forecast | 02 - Model Comparison vs Baseline

Four models are compared for each time series using a **time-ordered hold-out
set** (the last 20% of days) and grid search with `TimeSeriesSplit`:

| model | description |
|---|---|
| baseline | `DummyRegressor` - always predicts the training mean |
| linear | Ridge regression |
| random_forest | Random forest regressor |
| gradient_boosting | Gradient boosting regressor |

The same code (`model.compare_models`) is used by `model.py` when training
production models, so this notebook reflects exactly what gets deployed.
"""),
    code(SETUP),
    code("""from cslib import fetch_ts, engineer_features, DATA_DIR
from model import compare_models
ts = fetch_ts(DATA_DIR)
X, y, dates = engineer_features(ts['all'], training=True)
print(X.shape)
X.head()"""),
    md("## 1. Compare models on all countries combined"),
    code("""results, fitted = compare_models(X, y)
results"""),
    md("## 2. Visual comparison against the baseline (RMSE)"),
    code("""colors = ['tab:red' if m == 'baseline' else 'tab:blue' for m in results['model']]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(results['model'], results['rmse'], color=colors)
for i, v in enumerate(results['rmse']):
    ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
ax.set_title('Hold-out RMSE by model (all countries) - baseline in red')
ax.set_ylabel('RMSE (lower is better)')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'model_vs_baseline_rmse.png'), dpi=120)
plt.show()"""),
    md("## 3. Predictions vs actual on the hold-out period"),
    code("""split = int(len(X) * 0.8)
X_test, y_test, d_test = X.iloc[split:], y[split:], dates[split:]
best_name = results.loc[0, 'model']
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(d_test, y_test, color='black', lw=2, label='actual')
ax.plot(d_test, fitted[best_name].predict(X_test), color='tab:blue', label=f'best model: {best_name}')
ax.plot(d_test, fitted['baseline'].predict(X_test), color='tab:red', ls='--', label='baseline')
ax.set_title('Revenue for the next 30 days: best model vs baseline (hold-out)')
ax.set_ylabel('revenue'); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'model_vs_baseline_holdout.png'), dpi=120)
plt.show()"""),
    md("## 4. Improvement over the baseline for every country\n\nRead from `models/model_comparison.csv`, written by `python model.py`."),
    code("""comp = pd.read_csv(os.path.join('..', 'models', 'model_comparison.csv'))
pivot = comp.pivot(index='country', columns='model', values='rmse')
improvement = (1 - pivot.div(pivot['baseline'], axis=0)).drop(columns='baseline') * 100
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.heatmap(improvement, annot=True, fmt='.0f', cmap='RdBu', center=0, vmin=-100, vmax=100, ax=ax,
            cbar_kws={'label': '% RMSE improvement over baseline'})
ax.set_title('% RMSE improvement over the baseline (positive = better)')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'model_vs_baseline_by_country.png'), dpi=120)
plt.show()
comp.sort_values('rmse').groupby('country').first()[['model', 'rmse']]"""),
    md("""
## 5. Conclusions

* For all countries combined and for the largest country, the tree-based /
  linear models clearly beat the baseline; this supports the hypothesis from
  the EDA.
* For small countries with very noisy revenue, the baseline is sometimes the
  best choice. Because `model.py` always selects the model with the lowest
  hold-out RMSE, the deployed model for those countries is never worse than
  the baseline.
* Post-production, `monitoring.py` checks for data drift and compares the
  production RMSE with the training RMSE and the baseline.
"""),
]


def write(cells, name):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = os.path.join(NB_DIR, name)
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    os.makedirs(NB_DIR, exist_ok=True)
    paths = [write(eda, "01-EDA.ipynb"), write(model_nb, "02-model-comparison.ipynb")]
    if "--execute" in sys.argv:
        from nbconvert.preprocessors import ExecutePreprocessor
        for p in paths:
            nb = nbf.read(p, as_version=4)
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": NB_DIR}})
            nbf.write(nb, p)
            print("executed", p)

        # also export to Markdown + PNG files: GitHub sometimes fails to
        # display large .ipynb files, but always displays Markdown
        from nbconvert import MarkdownExporter
        from nbconvert.writers import FilesWriter
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            body, resources = MarkdownExporter().from_filename(
                p, resources={"output_files_dir": f"{name}_files"})
            FilesWriter(build_directory=NB_DIR).write(body, resources, notebook_name=name)
            print("exported", os.path.join(NB_DIR, name + ".md"))
