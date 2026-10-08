"""Summaries for retraining.py and real_data.py outputs."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def retrain():
    d = pd.read_csv("results/retrain.csv")
    base = d[d.trigger == "never"][["seed", "scenario", "eu_mean", "pr_mean", "brier_mean"]].rename(
        columns={"eu_mean": "eu0", "pr_mean": "pr0", "brier_mean": "br0"})
    x = d.merge(base, on=["seed", "scenario"])
    x["d_eu"] = x.eu_mean - x.eu0; x["d_pr"] = x.pr_mean - x.pr0; x["d_brier"] = x.brier_mean - x.br0
    rows = []
    for (sc, ru, tr, m), g in x.groupby(["scenario", "rule", "trigger", "delta_mult"]):
        lo, hi = boot_ci(g.d_eu)
        rows.append(dict(scenario=sc, rule=ru, trigger=tr, delta_mult=m, cand_rate=(g.n_candidates > 0).mean(),
                         promo_per_run=g.n_promotions.mean(), d_eu=g.d_eu.mean(), d_eu_lo=lo, d_eu_hi=hi,
                         d_eu_pct=100 * g.d_eu.mean() / g.eu0.mean(), d_pr=g.d_pr.mean(), d_brier=g.d_brier.mean()))
    s = pd.DataFrame(rows); s.to_csv("results/retrain_summary.csv", index=False)
    print("=== Retraining at delta x1 (EU change vs never, thousand units per slice, mean over drift slices) ===")
    print(s[(s.delta_mult == 1.0) & (s.trigger != "never")].round(3).to_string(index=False))
    print("\n=== Delta sensitivity: every_slice (promotions per run and EU change) ===")
    t = s[s.trigger == "every_slice"].pivot_table(index=["rule", "delta_mult"], columns="scenario", values=["promo_per_run", "d_eu"])
    print(t.round(2).to_string())


def real():
    d = pd.read_csv("results/real_pairs.csv")
    sig = ["horizon", "psi_score", "ks_score", "d_mean_p", "d_exp_brier", "d_u_mond", "psi_feat_max"]
    deg = ["deg_pr_auc", "deg_roc_auc", "deg_lift", "deg_reliability", "deg_ece", "deg_cal_slope", "deg_brier"]
    for ds, g in d.groupby("dataset"):
        g = g[g.horizon >= 1]
        print(f"\n=== {ds}: Spearman rho between label-free signal and realised degradation (n={len(g)} origin-horizon pairs) ===")
        tab = pd.DataFrame({s_: [spearmanr(g[s_], g[k]).statistic for k in deg] for s_ in sig}, index=deg).T
        print(tab.round(2).to_string())
        # cluster bootstrap CI over origins for psi_score vs reliability / pr
        origins = g.origin.unique(); rng = np.random.default_rng(1)
        for s_ in ["psi_score", "d_u_mond", "horizon"]:
            for k in ["deg_reliability", "deg_pr_auc"]:
                bs = []
                for _ in range(1000):
                    o = rng.choice(origins, len(origins)); gg = pd.concat([g[g.origin == oo] for oo in o])
                    bs.append(spearmanr(gg[s_], gg[k]).statistic)
                print(f"  {s_:10s} ~ {k:16s}: rho {spearmanr(g[s_], g[k]).statistic:+.2f}  95% CI [{np.nanpercentile(bs, 2.5):+.2f}, {np.nanpercentile(bs, 97.5):+.2f}]")
        o0 = g.origin.min(); a = d[(d.dataset == ds) & (d.origin == o0)]
        print(f"  trajectory for origin {o0}:")
        print(a[["month", "horizon", "base", "pr_auc", "roc_auc", "reliability", "ece", "cal_slope", "lift", "cov_mond_pos",
                 "psi_score", "d_u_mond", "psi_feat_max"]].round(4).to_string(index=False))
        print("  mean by horizon:")
        print(g.groupby("horizon")[["deg_pr_auc", "deg_reliability", "deg_ece", "psi_score", "d_u_mond"]].mean().round(4).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("retrain", "both") and os.path.exists("results/retrain.csv"):
        retrain()
    if what in ("real", "both") and os.path.exists("results/real_pairs.csv"):
        real()
