# AAVAIL Revenue Forecast | 01 - Exploratory Data Analysis (EDA)

**Business question:** AAVAIL management wants to predict the revenue for the
following month, for any country or for all countries combined, so staff can
plan resources.

**Data:** monthly JSON invoice files in `data/cs-train`. Ingestion is automated
by `cslib.fetch_data()` (raw transactions) and `cslib.fetch_ts()` (daily time
series per country). This notebook uses those same functions.


```python
import os, sys
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
FIG_DIR = os.path.join('..', 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
%matplotlib inline
```

## 1. Ingest the data


```python
from cslib import fetch_data, fetch_ts, DATA_DIR
df = fetch_data(DATA_DIR)
print(df.shape)
print('date range:', df.invoice_date.min().date(), 'to', df.invoice_date.max().date())
print('countries:', df.country.nunique())
df.head()
```

    (74207, 10)
    date range: 2017-11-28 to 2019-07-31
    countries: 19





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>customer_id</th>
      <th>day</th>
      <th>invoice</th>
      <th>month</th>
      <th>price</th>
      <th>stream_id</th>
      <th>times_viewed</th>
      <th>year</th>
      <th>invoice_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Norway</td>
      <td>13208.0</td>
      <td>28</td>
      <td>489001</td>
      <td>11</td>
      <td>3.01</td>
      <td>22869</td>
      <td>19</td>
      <td>2017</td>
      <td>2017-11-28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>France</td>
      <td>13726.0</td>
      <td>28</td>
      <td>489028</td>
      <td>11</td>
      <td>1.69</td>
      <td>23252</td>
      <td>18</td>
      <td>2017</td>
      <td>2017-11-28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>France</td>
      <td>13726.0</td>
      <td>28</td>
      <td>489028</td>
      <td>11</td>
      <td>2.04</td>
      <td>20399</td>
      <td>18</td>
      <td>2017</td>
      <td>2017-11-28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>France</td>
      <td>13726.0</td>
      <td>28</td>
      <td>489028</td>
      <td>11</td>
      <td>0.90</td>
      <td>22036</td>
      <td>2</td>
      <td>2017</td>
      <td>2017-11-28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>28</td>
      <td>489029</td>
      <td>11</td>
      <td>5.71</td>
      <td>23535</td>
      <td>9</td>
      <td>2017</td>
      <td>2017-11-28</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Data quality: missing values and summary statistics


```python
missing = df.isnull().mean().mul(100).round(2).rename('% missing')
display(missing.to_frame())
df[['price', 'times_viewed']].describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>% missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>country</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>customer_id</th>
      <td>20.13</td>
    </tr>
    <tr>
      <th>day</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>invoice</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>price</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>stream_id</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>times_viewed</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>invoice_date</th>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>times_viewed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>74207.000000</td>
      <td>74207.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.769427</td>
      <td>12.502581</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.656977</td>
      <td>6.926683</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.100000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.590000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.720000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.660000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>149.270000</td>
      <td>24.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Revenue over time (monthly)


```python
monthly = df.set_index('invoice_date')['price'].resample('MS').sum()
days_per_month = df.groupby(df.invoice_date.dt.to_period('M')).invoice_date.nunique()
monthly = monthly[days_per_month.values >= 20]  # drop partial months at the edges
fig, ax = plt.subplots(figsize=(11, 4))
monthly.plot(ax=ax, marker='o')
ax.set_title('Total monthly revenue (all countries)')
ax.set_ylabel('revenue'); ax.set_xlabel('')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_monthly_revenue.png'), dpi=120)
plt.show()
```


    
![png](01-EDA_files/output_7_0.png)
    


## 4. Top 10 countries by revenue


```python
by_country = df.groupby('country')['price'].sum().sort_values(ascending=False)
top10 = by_country.head(10)
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(x=top10.values, y=top10.index, ax=ax, color='tab:blue')
ax.set_title('Top 10 countries by total revenue'); ax.set_xlabel('revenue'); ax.set_ylabel('')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_top10_countries.png'), dpi=120)
plt.show()
print(f"Top 10 countries = {top10.sum() / by_country.sum():.1%} of revenue")
```


    
![png](01-EDA_files/output_9_0.png)
    


    Top 10 countries = 93.9% of revenue


## 5. Daily revenue and 30-day rolling average for the top countries


```python
ts = fetch_ts(DATA_DIR)
keys = ['all'] + [k for k in ts if k != 'all'][:5]
fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.3 * len(keys)), sharex=True)
for ax, key in zip(axes, keys):
    s = ts[key].set_index('date')['revenue']
    ax.plot(s.index, s.values, alpha=0.35, lw=0.8)
    ax.plot(s.index, s.rolling(30).mean(), lw=2)
    ax.set_title(key, loc='left', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_daily_revenue_by_country.png'), dpi=120)
plt.show()
```


    
![png](01-EDA_files/output_11_0.png)
    


## 6. Weekly pattern


```python
daily = ts['all'].copy()
daily['weekday'] = daily['date'].dt.day_name()
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
fig, ax = plt.subplots(figsize=(9, 4))
sns.boxplot(data=daily, x='weekday', y='revenue', order=order, ax=ax, color='tab:blue')
ax.set_title('Daily revenue by day of week'); ax.set_xlabel('')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_weekday.png'), dpi=120)
plt.show()
```


    
![png](01-EDA_files/output_13_0.png)
    


## 7. Relationships between daily features


```python
corr = ts['all'][['purchases','unique_invoices','unique_streams','total_views','revenue']].corr()
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax)
ax.set_title('Correlation of daily features')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_correlation.png'), dpi=120)
plt.show()
```


    
![png](01-EDA_files/output_15_0.png)
    


## 8. Distribution of daily revenue


```python
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(ts['all']['revenue'], bins=40, kde=True, ax=ax)
ax.set_title('Distribution of daily revenue (all countries)')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'eda_revenue_distribution.png'), dpi=120)
plt.show()
```


    
![png](01-EDA_files/output_17_0.png)
    


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
