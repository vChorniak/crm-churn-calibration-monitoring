# Data

Raw and processed data are **not** included in this repository. The synthetic experiments need no data.
For the real-data analyses, download the two public datasets and place them as follows:

```
data/raw/kkbox/transactions.csv
data/raw/kkbox/transactions_v2.csv
data/raw/kkbox/members_v3.csv
data/raw/kkbox/train.csv
data/raw/online_retail_II.xlsx
```

* **KKBox** — WSDM Cup 2018, Kaggle competition "WSDM – KKBox's Churn Prediction Challenge"
  (https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge). Downloading requires a Kaggle account
  and acceptance of the competition rules; the files are distributed under those rules and therefore are not
  redistributed here. Only the four files above are needed (`user_logs*.csv` are not used).
  `transactions_v2.csv` and `train.csv` are unpacked from the `*.7z` archives of the same names.
* **Online Retail II** — UCI Machine Learning Repository, https://doi.org/10.24432/C5CG6D (CC BY 4.0).

`run_all.sh` converts them into `data/processed/kkbox_monthly.parquet`, `data/processed/kkbox_month_stats.csv`
and `data/processed/retail_monthly.parquet`. Another location can be used by setting `DATA_DIR`.
