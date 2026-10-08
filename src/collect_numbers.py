"""Collect every number quoted in the manuscript from the result files (single source of truth)."""
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

import os
R = "results/"
DATA = os.environ.get("DATA_DIR", "data/processed").rstrip("/") + "/"
N = {}
f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"; f1 = lambda x: f"{x:.1f}"; f4 = lambda x: f"{x:.4f}"
pct = lambda x: f"{100 * x:.0f}%"

d = pd.concat([pd.read_csv(R + "slices.csv"), pd.read_csv(R + "slices_extra.csv")], ignore_index=True)
st = d[(d.scenario == "none") & (d.slice <= 12)]
for k in ["base", "roc_auc", "pr_auc", "brier", "ece", "lift", "eu_decile", "cov_marg", "cov_marg_pos", "cov_mond", "cov_mond_pos", "cov_mond_neg", "u_mond", "u_marg"]:
    N[f"st_{k}"] = st[k].mean(); N[f"st_{k}_sd"] = st.groupby("seed")[k].mean().std(ddof=1)
N["st_brier_ref"] = (st.base * (1 - st.base)).mean()
N["corr_umarg_share"] = np.corrcoef(d.u_marg, d.share_p_gt_qmarg)[0, 1]

# drift harm at slice 18 vs stable
harm = {}
for sc, g in d.groupby("scenario"):
    s0, s1 = g[g.slice <= 12], g[g.slice == 18]
    harm[sc] = dict(base0=s0.base.mean(), base1=s1.base.mean(), pr0=s0.pr_auc.mean(), pr1=s1.pr_auc.mean(),
                    rel0=s0.reliability.mean(), rel1=s1.reliability.mean(), eu0=s0.eu_decile.mean(), eu1=s1.eu_decile.mean(),
                    roc0=s0.roc_auc.mean(), roc1=s1.roc_auc.mean(), cal1=s1.cal_slope.mean(),
                    nth0=s0.n_thresh.mean(), nth1=s1.n_thresh.mean(), eut0=s0.eu_thresh.mean(), eut1=s1.eu_thresh.mean(),
                    covpos1=s1.cov_mond_pos.mean(), covneg1=s1.cov_mond_neg.mean())
N["harm"] = harm

det = pd.read_csv(R + "detection.csv"); ds = pd.read_csv(R + "detection_summary.csv")
N["det"] = ds.to_dict("records")
c = det[(det.scenario == "combined_prior") & (det.rule == "3sigma")].pivot(index="seed", columns="signal", values="delay")
def pair(a, b):
    diff = c[a] - c[b]
    p = wilcoxon(c[a], c[b]).pvalue if (diff != 0).any() else 1.0
    return dict(diff=diff.mean(), earlier=(diff < 0).mean(), ties=(diff == 0).mean(), p=p)
N["pair_rel_pr"] = pair("reliability", "pr_auc"); N["pair_brier_pr"] = pair("brier", "pr_auc")
N["pair_brier_roc"] = pair("brier", "roc_auc"); N["pair_u_meanp"] = pair("u_mond", "mean_p"); N["pair_u_psi"] = pair("u_mond", "psi_score")
fpr = ds[(ds.scenario == "none")]
N["fpr_3s_min"], N["fpr_3s_max"], N["fpr_3s_mean"] = [fpr[fpr.rule == "3sigma"].det_rate.agg(a) for a in ("min", "max", "mean")]
N["fpr_2of3_mean"] = fpr[fpr.rule == "2of3"].det_rate.mean()

info = pd.read_csv(R + "model_info.csv")
N["l1_zero_pure"] = info.zero_pure_noise.mean(); N["l1_zero_corr"] = info.zero_corr_noise.mean(); N["l1_zero_core"] = info.zero_core.mean()

rt = pd.read_csv(R + "retrain_summary.csv"); N["retrain"] = rt.to_dict("records")
nz = pd.read_csv(R + "noise.csv"); N["noise"] = nz.groupby(["noise", "learner"]).mean(numeric_only=True).reset_index().to_dict("records")

rp = pd.read_csv(R + "real_pairs.csv"); g = rp[rp.horizon >= 1]
sp = {}
for dsn, gg in g.groupby("dataset"):
    sp[dsn] = {s: {k: spearmanr(gg[s], gg[k]).statistic for k in ["deg_reliability", "deg_ece", "deg_pr_auc", "deg_roc_auc"]}
               for s in ["horizon", "psi_score", "ks_score", "d_mean_p", "d_u_mond", "psi_feat_max"]}
    sp[dsn]["n_pairs"] = len(gg)
    h0 = rp[(rp.dataset == dsn) & (rp.horizon == 0)]
    sp[dsn]["cov_pos_h0"] = h0.cov_mond_pos.mean(); sp[dsn]["cov_pos_h"] = gg.cov_mond_pos.mean()
    sp[dsn]["cov_pos_min"] = gg.cov_mond_pos.min()
    sp[dsn]["psi_feat_med"] = gg.psi_feat_max.median()
    origins = gg.origin.unique(); rng = np.random.default_rng(1); ci = {}
    for s_ in ["psi_score", "d_u_mond", "horizon"]:
        bs_ = []
        for _ in range(1000):
            o = rng.choice(origins, len(origins)); x = pd.concat([gg[gg.origin == oo] for oo in o])
            bs_.append(spearmanr(x[s_], x.deg_reliability).statistic)
        ci[s_] = [float(np.nanpercentile(bs_, 2.5)), float(np.nanpercentile(bs_, 97.5))]
    sp[dsn]["ci_rel"] = ci
N["real_spearman"] = sp
try:
    N["real_cluster_ci"] = json.load(open(R + "real_cluster_ci.json"))
except FileNotFoundError:
    pass
ms = pd.read_csv(DATA + "kkbox_month_stats.csv")
N["kk_months"] = len(ms); N["kk_rate_min"] = ms.churn_rate.min(); N["kk_rate_max"] = ms.churn_rate.max()
N["kk_cand_min"] = int(ms.n_candidates.min()); N["kk_cand_max"] = int(ms.n_candidates.max())
rr = pd.read_parquet(DATA + "retail_monthly.parquet")
a = rr.groupby("cutoff").agg(n=("cid", "size"), r=("is_churn", "mean"))
N["rt_slices"] = len(a); N["rt_n_min"] = int(a.n.min()); N["rt_n_max"] = int(a.n.max()); N["rt_rate_min"] = a.r.min(); N["rt_rate_max"] = a.r.max()
try:
    b = pd.read_csv(R + "baselines_synth.csv"); bs = b[(b.scenario == "none") & (b.slice <= 12)]
    N["base_stable"] = bs.groupby("model")[["pr_auc", "roc_auc", "brier", "ece", "lift", "eu_decile"]].mean().reset_index().to_dict("records")
    bd = pd.read_csv(R + "baselines_synth_detect.csv")
    N["base_detect"] = bd.groupby(["scenario", "model", "signal"]).delay.agg(["mean", lambda x: (x <= 6).mean()]).reset_index().rename(columns={"<lambda_0>": "rate"}).to_dict("records")
    br = pd.read_csv(R + "baselines_real.csv")
    br["h"] = np.where(br.horizon == 0, "in-time", "h1-6")
    N["base_real"] = br.groupby(["dataset", "model", "h"])[["pr_auc", "roc_auc", "brier", "reliability", "ece", "lift"]].mean().reset_index().to_dict("records")
except FileNotFoundError:
    pass
json.dump(N, open(R + "numbers.json", "w"), indent=1, default=float)
print("ok", len(N))
