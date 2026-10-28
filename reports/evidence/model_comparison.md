# AAVAIL Revenue Forecast | Model comparison

Four models were compared for every country on a time-ordered hold-out set
(last 20% of days). Values are RMSE of 30-day revenue (lower is better).
The model with the lowest RMSE is saved and served by the API.

| country | baseline | linear | random_forest | gradient_boosting | selected_model |
|---|---|---|---|---|---|
| all | 2967.1 | 1320.0 | 1030.7 | 1036.9 | random_forest |
| eire | 234.5 | 249.5 | 242.5 | 236.7 | baseline |
| france | 120.9 | 113.5 | 134.3 | 123.8 | linear |
| germany | 191.1 | 232.5 | 186.8 | 189.6 | random_forest |
| hong_kong | 94.3 | 90.6 | 132.7 | 157.9 | linear |
| netherlands | 70.5 | 110.1 | 131.6 | 159.2 | baseline |
| norway | 169.2 | 108.1 | 168.2 | 155.9 | linear |
| portugal | 78.5 | 141.5 | 202.4 | 178.1 | baseline |
| singapore | 40.7 | 79.2 | 194.5 | 172.9 | baseline |
| spain | 69.2 | 82.4 | 78.2 | 109.0 | baseline |
| united_kingdom | 1959.9 | 947.6 | 953.4 | 1005.0 | linear |

Charts: `reports/figures/model_vs_baseline_rmse.png`, `model_vs_baseline_holdout.png`, `model_vs_baseline_by_country.png`.

