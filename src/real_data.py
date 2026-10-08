"""
Real-data study: KKBox (subscription churn) and Online Retail II (retail churn).

Rolling-origin design: for every origin month o the full pipeline (class-weighted L1-LR + isotonic +
Mondrian conformal) is fitted on month o (60 % train / 20 % calibration / 20 % in-time test) and then
applied, unchanged, to months o+1 ... o+H. For every (origin, horizon) pair we record
  * label-free monitoring signals (computed from features and scores only),
  * realised quality (needs labels): PR-AUC, ROC-AUC, Brier, Murphy reliability, ECE, calibration slope,
    Lift@10 %, conformal coverage,
  * degradation relative to the in-time test part of the origin month.
Run: python real_data.py -> results/real_pairs.csv
"""
import sys, csv, warnings
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score
from simulation import Model, psi, murphy, cal_slope, lift

warnings.filterwarnings("ignore")
import os
UP = os.environ.get("DATA_DIR", "data/processed").rstrip("/") + "/"
H = 6


def kkbox():
    d = pd.read_parquet(UP + "kkbox_monthly.parquet")
    num = ["payment_plan_days", "plan_list_price", "actual_amount_paid", "last_discount", "is_auto_renew",
           "last_is_cancel", "days_to_expire", "recency_days", "tx_tenure_days", "n_tx_total", "n_tx_365",
           "n_cancel_365", "amount_365", "share_auto_365", "mean_discount_365", "n_distinct_plans"]
    X = d[num].astype(float).copy()
    for c in ["tx_tenure_days", "n_tx_total", "n_tx_365", "n_cancel_365", "amount_365"]:
        X[c] = np.log1p(X[c].clip(lower=0))
    X["member_missing"] = d.city.isna().astype(float)
    X["age"] = d.age.fillna(0); X["age_missing"] = d.age.isna().astype(float)
    X["reg_tenure"] = np.log1p(d.reg_tenure_days.clip(lower=0).fillna(0))
    for col, cats in [("payment_method_id", [41, 40, 38, 37, 34, 36]), ("city", [1, 13, 5, 4]),
                      ("registered_via", [7, 9, 3, 4]), ("gender", ["male", "female"])]:
        for c in cats:
            X[f"{col}_{c}"] = (d[col] == c).astype(float)
    months = sorted(d.month.unique())
    key = d.month.map({m: i for i, m in enumerate(months)}).values
    return "kkbox", X.values, d.is_churn.values, key, [str(m) for m in months], list(X.columns), 1


def retail():
    d = pd.read_parquet(UP + "retail_monthly.parquet")
    num = ["recency_days", "tenure_days", "n_inv_365", "n_inv_90", "amount_365", "amount_90", "avg_basket",
           "n_products_365", "n_active_months_365", "n_returns_365", "return_amount_365", "uk"]
    X = d[num].astype(float).copy()
    for c in num[:-1]:
        X[c] = np.log1p(X[c].clip(lower=0))
    months = sorted(d.cutoff.astype(str).unique())
    key = d.cutoff.astype(str).map({m: i for i, m in enumerate(months)}).values
    return "retail", X.values, d.is_churn.values, key, months, num, 3


def quality(model, X, y):
    p = model.predict(X)
    i0, i1 = model.sets(p, mondrian=True)
    rel, res, ece = murphy(y, p, bins=10 if len(y) < 20000 else 20)
    sl, _ = cal_slope(y, p)
    return p, dict(base=y.mean(), pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                   brier=np.mean((p - y) ** 2), reliability=rel, ece=ece, cal_slope=sl, lift=lift(y, p),
                   cov_mond=np.where(y == 1, i1, i0).mean(), cov_mond_pos=i1[y == 1].mean())


def label_free(model, X, p, ref_X, ref_p, ref):
    i0, i1 = model.sets(p, mondrian=True)
    return dict(psi_score=psi(ref_p, p), ks_score=ks_2samp(ref_p, p).statistic,
                d_mean_p=abs(p.mean() - ref["mean_p"]), d_exp_brier=abs(np.mean(p * (1 - p)) - ref["exp_brier"]),
                d_u_mond=abs(((i0.astype(int) + i1) != 1).mean() - ref["u_mond"]),
                psi_feat_max=max(psi(ref_X[:, j], X[:, j]) for j in range(X.shape[1]) if len(np.unique(ref_X[:, j])) > 2))


def study(ds, origins, rows):
    name, X, y, key, months, cols, lag = ds
    for o in origins:
        idx = np.where(key == o)[0]
        rng = np.random.default_rng(o); idx = rng.permutation(idx)
        a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
        tr, ca, te = idx[:a], idx[a:b], idx[b:]
        m = Model(X[tr], y[tr], X[ca], y[ca])
        p_te, q0 = quality(m, X[te], y[te])
        ref_p = m.predict(X[ca])
        i0, i1 = m.sets(ref_p, mondrian=True)
        ref = dict(mean_p=ref_p.mean(), exp_brier=np.mean(ref_p * (1 - ref_p)), u_mond=((i0.astype(int) + i1) != 1).mean())
        for h in range(0, H + 1):
            t = o + h
            if t >= len(months):
                break
            sel = te if h == 0 else np.where(key == t)[0]
            p, q = quality(m, X[sel], y[sel])
            lf = label_free(m, X[sel], p, X[ca], ref_p, ref)
            r = dict(dataset=name, origin=months[o], month=months[t], horizon=h, label_lag=lag, n=len(sel))
            r.update(q); r.update(lf)
            r.update({f"deg_{k}": q0[k] - q[k] for k in ["pr_auc", "roc_auc", "lift"]})
            r.update({f"deg_{k}": q[k] - q0[k] for k in ["reliability", "ece", "brier"]})
            r["deg_cal_slope"] = abs(q["cal_slope"] - 1) - abs(q0["cal_slope"] - 1)
            rows.append(r)
        print(name, months[o], "done", flush=True)


if __name__ == "__main__":
    rows = []
    kk = kkbox()
    study(kk, [kk[4].index(m) for m in ["201601", "201602", "201603", "201604", "201605", "201606", "201607", "201608"]], rows)
    rt = retail()
    study(rt, list(range(0, 12)), rows)
    pd.DataFrame(rows).to_csv("results/real_pairs.csv", index=False)
    print("done", len(rows))
