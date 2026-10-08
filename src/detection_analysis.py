"""Detection-time analysis for simulation.py results -> results/detection.csv, results/detection_summary.csv and a printed summary."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

S, D = 12, 6
SIGNALS = [  # name, direction(+1 up is bad, -1 down is bad, 0 two-sided), needs labels
    ("brier", 1, True), ("reliability", 1, True), ("ece", 1, True), ("cal_slope", 0, True),
    ("pr_auc", -1, True), ("roc_auc", -1, True), ("lift", -1, True), ("accuracy", -1, True),
    ("cov_marg", -1, True), ("cov_mond", -1, True), ("eu_decile", -1, True),
    ("mean_p", 0, False), ("exp_brier", 0, False), ("u_marg", 0, False), ("u_mond", 0, False),
    ("psi_score", 1, False), ("ks_score", 1, False), ("psi_feat_max", 1, False),
]
FIXED = [("psi_score>0.10", "psi_score", 0.10), ("psi_score>0.25", "psi_score", 0.25), ("psi_feat_max>0.10", "psi_feat_max", 0.10)]


def z_scores(v, direction):
    mu, sd = v[:S].mean(), v[:S].std(ddof=1)
    z = (v - mu) / (sd if sd > 0 else 1e-12)
    return z * direction if direction else np.abs(z)


def first(flags):
    idx = np.where(flags[S:])[0]
    return int(idx[0]) + 1 if len(idx) else D + 1       # delay in slices after drift onset (1 = first drift slice)


def detect(v, direction, rule):
    z = z_scores(v, direction)
    if rule == "3sigma":
        return first(z > 3)
    # 2 of 3 consecutive beyond 2 sigma (same side)
    b = z > 2
    flags = np.array([i >= 2 and b[i - 2:i + 1].sum() >= 2 and b[i] for i in range(len(b))])
    return first(flags)


def main():
    import os
    d = pd.read_csv("results/slices.csv")
    if os.path.exists("results/slices_extra.csv"):
        d = pd.concat([d, pd.read_csv("results/slices_extra.csv")], ignore_index=True)
    rows = []
    for (seed, scen), g in d.groupby(["seed", "scenario"]):
        g = g.sort_values("slice")
        for name, direc, lab in SIGNALS:
            for rule in ("3sigma", "2of3"):
                rows.append(dict(seed=seed, scenario=scen, signal=name, rule=rule, labels=lab,
                                 delay=detect(g[name].values, direc, rule)))
        for name, col, thr in FIXED:
            rows.append(dict(seed=seed, scenario=scen, signal=name, rule="fixed", labels=False,
                             delay=first(g[col].values > thr)))
    r = pd.DataFrame(rows)
    r["detected"] = r.delay <= D
    r["cal_delay"] = r.delay + r.labels.astype(int)      # labels arrive one slice later
    r.to_csv("results/detection.csv", index=False)
    rng = np.random.default_rng(0)

    def ci(x):
        b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        return np.percentile(b, [2.5, 97.5])

    out = []
    for (scen, sig, rule), g in r.groupby(["scenario", "signal", "rule"]):
        lo, hi = ci(g.delay.values)
        out.append(dict(scenario=scen, signal=sig, rule=rule, labels=g.labels.iloc[0], det_rate=g.detected.mean(),
                        mean_delay=g.delay.mean(), ci_lo=lo, ci_hi=hi, mean_cal_delay=g.cal_delay.mean()))
    s = pd.DataFrame(out)
    s.to_csv("results/detection_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    fpr = s[s.scenario == "none"][["signal", "rule", "det_rate"]].rename(columns={"det_rate": "FPR_6slices"})
    for scen in ["combined", "combined_prior", "covariate", "concept", "abrupt"]:
        t = s[(s.scenario == scen) & (s.rule.isin(["3sigma", "fixed"]))].merge(
            fpr[fpr.rule.isin(["3sigma", "fixed"])], on=["signal", "rule"]).sort_values("mean_cal_delay")
        print(f"\n=== {scen} (3-sigma / fixed) ===")
        print(t[["signal", "rule", "labels", "det_rate", "mean_delay", "ci_lo", "ci_hi", "mean_cal_delay", "FPR_6slices"]].round(3).to_string(index=False))
    print("\n=== 2-of-3 rule, combined + FPR ===")
    t = s[(s.scenario == "combined") & (s.rule == "2of3")].merge(fpr[fpr.rule == "2of3"], on=["signal", "rule"]).sort_values("mean_cal_delay")
    print(t[["signal", "labels", "det_rate", "mean_delay", "mean_cal_delay", "FPR_6slices"]].round(3).to_string(index=False))
    # paired tests (combined, 3sigma)
    for scen_t in ["combined", "combined_prior"]:
        paired(r, scen_t)
    harm(d)


def paired(r, scen):
    c = r[(r.scenario == scen) & (r.rule == "3sigma")].pivot(index="seed", columns="signal", values="delay")
    print(f"\nPaired Wilcoxon ({scen}, 3sigma, delay):")
    for a, b in [("brier", "pr_auc"), ("brier", "roc_auc"), ("reliability", "pr_auc"), ("u_mond", "mean_p"),
                 ("u_mond", "psi_score"), ("u_mond", "brier"), ("mean_p", "psi_score")]:
        diff = c[a] - c[b]
        p = wilcoxon(c[a], c[b]).pvalue if (diff != 0).any() else 1.0
        print(f"  {a:12s} vs {b:12s}: mean diff {diff.mean():+.2f}, {a} earlier in {(diff < 0).mean():.0%}, ties {(diff == 0).mean():.0%}, p={p:.3g}")


def harm(d):
    # harmfulness of each drift type (slice 18 vs stable mean)
    st = d[d.slice <= S].groupby("scenario")[["pr_auc", "reliability", "eu_decile", "lift", "base"]].mean()
    en = d[d.slice == S + D].groupby("scenario")[["pr_auc", "reliability", "eu_decile", "lift", "base"]].mean()
    print("\nEffect of drift at last slice (value, relative change):")
    print(pd.concat([en.round(4), ((en / st - 1) * 100).round(1).add_suffix("_%")], axis=1).to_string())
    info = pd.read_csv("results/model_info.csv")
    print("\nL1 zeroed (mean of 50): core %.2f/7, corr-noise %.2f/5, pure-noise %.2f/5" %
          (info.zero_core.mean(), info.zero_corr_noise.mean(), info.zero_pure_noise.mean()))
    st2 = d[(d.slice <= S)]
    print("Stable: cov_marg %.3f (pos %.3f), cov_mond %.3f (pos %.3f, neg %.3f), U_marg %.3f, U_mond %.3f (both %.3f, empty %.3f)" % tuple(
        st2[c].mean() for c in ["cov_marg", "cov_marg_pos", "cov_mond", "cov_mond_pos", "cov_mond_neg", "u_marg", "u_mond", "u_mond_both", "u_mond_empty"]))
    print("corr(U_mond, mean_p) over all slices: %.3f; corr(U_marg, share p>q): %.4f" % (
        np.corrcoef(d.u_mond, d.mean_p)[0, 1], np.corrcoef(d.u_marg, d.share_p_gt_qmarg)[0, 1]))


if __name__ == "__main__":
    main()
