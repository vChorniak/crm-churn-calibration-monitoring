"""Figures for the Applied Sciences manuscript (300 dpi PNG, light background)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
C1, C2, C3, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})
S = 12


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png"); plt.close(fig); print("saved", name)


# ------------------------------------------------------------------ Fig 1: framework
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(-3, 46)
    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec="#c9c8c3", lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.3, body, ha="center", va="top", fontsize=7.2, color=INK2, linespacing=1.35)
    def arrow(a, b, txt=None, dy=1.2):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1))
        if txt:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, txt, ha="center", fontsize=6.8, color=INK2)
    box(1, 26, 19, 18, "Scoring", "L1 logistic model\n+ isotonic recalibration\n+ Mondrian conformal\nsets", "#f4f8fd")
    box(27, 26, 20, 18, "Label-free monitors", "score PSI / KS, mean p̂,\nexpected Brier,\nconformal ambiguity U\n(available at time t)", "#fdf3ee")
    box(54, 26, 20, 18, "Label-based monitors", "reliability, ECE,\ncalibration slope, Brier,\nPR-AUC, ROC-AUC, lift\n(available at t + lag)", "#eef9f4")
    box(80, 26, 19, 18, "Decision layer", "threshold policy p̂ > c/b\nor fixed-coverage\n(top decile)", "#f4f3f0")
    box(27, 2, 20, 17, "Label collection", "early warning triggers\nlabel acquisition,\nnot immediate retraining", "#fdf3ee")
    box(54, 2, 20, 17, "Candidate + promotion", "retrain on latest labelled\nslice; promote only if\nnon-inferior on all facets\nand better on one", "#eef9f4")
    arrow((20.8, 35), (26.2, 35)); arrow((47.8, 35), (53.2, 35)); arrow((74.8, 35), (79.2, 35))
    arrow((37, 25.2), (37, 19.8), "alarm", dy=0); arrow((47.8, 10.5), (53.2, 10.5)); ax.text(50.5, 12.2, "labels", ha="center", fontsize=6.8, color=INK2, bbox=dict(fc="white", ec="none", pad=0.5))
    arrow((64, 19.8), (64, 25.2), "alarm", dy=0)
    ax.plot([64, 64, 10.5], [1.6, -1.5, -1.5], color=INK2, lw=1)
    ax.add_patch(FancyArrowPatch((10.5, -1.5), (10.5, 25.2), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1))
    ax.text(12, 12, "promoted model\nreplaces the\nincumbent", fontsize=6.8, color=INK2)
    save(fig, "fig1_framework")


# ------------------------------------------------------------------ Fig 2: detection map
LABEL = {"reliability": "Reliability (Murphy)", "ece": "ECE", "cal_slope": "Calibration slope", "brier": "Brier score",
         "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "lift": "Lift@10%", "u_mond": "Conformal ambiguity U",
         "psi_score": "Score PSI (3σ)", "mean_p": "Mean score", "psi_score>0.10": "Score PSI > 0.10"}


def fig2():
    s = pd.read_csv("results/detection_summary.csv")
    sigs = ["reliability", "ece", "cal_slope", "brier", "pr_auc", "roc_auc", "lift", "u_mond", "psi_score", "mean_p", "psi_score>0.10"]
    scen = [("combined_prior", "Combined, prior fixed\n(harmful)"), ("concept", "Concept only\n(harmful)"),
            ("covariate", "Covariate only\n(benign)"), ("none", "No drift\n(false alarms)")]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.6), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1, 0.75]})
    y = np.arange(len(sigs))[::-1]
    for ax, (sc, title) in zip(axes, scen):
        ax.set_title(title, fontsize=8.5, color=INK)
        for yi, sg in zip(y, sigs):
            rule = "fixed" if ">" in sg else "3sigma"
            r = s[(s.scenario == sc) & (s.signal == sg) & (s.rule == rule)].iloc[0]
            col = C1 if r.labels else C2
            if sc == "none":
                ax.barh(yi, 100 * r.det_rate, height=0.5, color=col, alpha=0.85)
                ax.text(100 * r.det_rate + 0.8, yi, f"{100 * r.det_rate:.0f}%", va="center", fontsize=7, color=INK2)
                continue
            lab_lag = 1 if r.labels else 0
            if r.det_rate < 0.5:
                ax.text(4.5, yi, f"detected in {100 * r.det_rate:.0f}%", va="center", ha="center", fontsize=6.8, color=GREY)
                continue
            m, lo, hi = r.mean_delay + lab_lag, r.ci_lo + lab_lag, r.ci_hi + lab_lag
            ax.plot([lo, hi], [yi, yi], color=col, lw=2, alpha=0.5)
            ax.scatter([m], [yi], s=30, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        if sc == "none":
            ax.set_xlim(0, 20); ax.set_xlabel("false-alarm rate, %\n(6 monitored slices)")
        else:
            ax.set_xlim(0.5, 8.5); ax.set_xticks(range(1, 9)); ax.set_xlabel("calendar delay, slices")
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL[s_] for s_ in sigs])
    axes[0].scatter([], [], color=C1, label="needs labels (+1 slice lag)"); axes[0].scatter([], [], color=C2, label="label-free")
    axes[3].grid(axis="y", visible=False)
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.07), fontsize=8)
    save(fig, "fig2_detection_map")


# ------------------------------------------------------------------ Fig 3: trajectories (combined_prior)
def fig3():
    d = pd.concat([pd.read_csv("results/slices.csv"), pd.read_csv("results/slices_extra.csv")])
    d = d[d.scenario == "combined_prior"]
    def z(col, direc):
        out = []
        for _, g in d.groupby("seed"):
            v = g.sort_values("slice")[col].values; mu, sd = v[:S].mean(), v[:S].std(ddof=1)
            zz = (v - mu) / sd; out.append(zz * direc if direc else np.abs(zz))
        return np.median(np.array(out), axis=0)
    t = np.arange(1, 19)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, series, title in [(axes[0], [("reliability", 1, C1, "Reliability"), ("brier", 1, C2, "Brier"), ("pr_auc", -1, C3, "PR-AUC"), ("roc_auc", -1, GREY, "ROC-AUC")], "(a) label-based signals"),
                              (axes[1], [("u_mond", 0, C1, "Conformal U"), ("psi_score", 1, C2, "Score PSI"), ("mean_p", 0, C3, "Mean score")], "(b) label-free signals")]:
        ax.axvspan(12.5, 18.5, color="#f4f3f0", zorder=0); ax.axhline(3, color=INK2, lw=1)
        ax.text(7, 3.5, "3σ alarm limit", fontsize=7, color=INK2)
        for col, direc, c, name in series:
            zz = np.clip(z(col, direc), -3, 40)
            ax.plot(t, zz, color=c, label=name)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left"); ax.set_xlabel("slice (drift from slice 13)")
        ax.set_yscale("symlog", linthresh=3); ax.set_ylim(-3, 60); ax.legend(fontsize=7.5, loc="upper left")
        ax.set_yticks([-3, 0, 3, 10, 40]); ax.set_yticklabels(["−3", "0", "3", "10", "≥40"]); ax.set_xticks([1, 6, 12, 18])
    axes[0].set_ylabel("median standardised deviation\n(bad direction)")
    save(fig, "fig3_trajectories")


# ------------------------------------------------------------------ Fig 4: policy sensitivity
def fig4():
    d = pd.read_csv("results/slices_extra.csv"); d = d[d.scenario == "combined_prior"]
    g = d.groupby("slice")[["n_thresh", "eu_thresh", "eu_decile"]].mean(); t = g.index.values
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax in axes:
        ax.axvspan(12.5, 18.5, color="#f4f3f0", zorder=0); ax.set_xticks([1, 6, 12, 18]); ax.set_xlabel("slice (drift from slice 13)")
    axes[0].plot(t, g.n_thresh / 1000, color=C1, label="threshold policy p̂ > c/b")
    axes[0].plot(t, np.full(len(t), 5.0), color=C2, label="fixed top decile")
    axes[0].set_ylabel("targeted customers, thousands"); axes[0].set_ylim(0, 18); axes[0].legend(fontsize=7.5, loc="upper left")
    axes[0].set_title("(a) size of the targeted set", fontsize=8.5, loc="left", color=INK)
    axes[1].plot(t, g.eu_thresh, color=C1, label="threshold policy"); axes[1].plot(t, g.eu_decile, color=C2, label="fixed top decile")
    axes[1].set_ylabel("realised utility, thousand units"); axes[1].legend(fontsize=7.5, loc="lower left")
    axes[1].set_title("(b) realised utility", fontsize=8.5, loc="left", color=INK)
    save(fig, "fig4_policy")


# ------------------------------------------------------------------ Fig 5: real data
def fig5():
    d = pd.read_csv("results/real_pairs.csv"); d = d[d.horizon >= 1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, (ds, title) in zip(axes, [("kkbox", "(a) KKBox (subscription music)"), ("retail", "(b) Online Retail II")]):
        g = d[d.dataset == ds]
        ax.scatter(g.psi_score, g.deg_reliability * 1000, s=22, color=C1, edgecolor="white", linewidth=1, label="score PSI", zorder=3)
        rho = spearmanr(g.psi_score, g.deg_reliability).statistic; rh = spearmanr(g.horizon, g.deg_reliability).statistic
        ax.text(0.97, 0.95, f"Spearman ρ (PSI) = {rho:.2f}\nρ (model age) = {rh:.2f}", transform=ax.transAxes, va="top", ha="right", fontsize=7.5, color=INK2)
        ax.set_xlabel("score PSI vs. calibration sample"); ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    axes[0].set_ylabel("increase in reliability term, ×10⁻³")
    save(fig, "fig5_real")


# ------------------------------------------------------------------ Fig 6: promotion rule
def fig6():
    s = pd.read_csv("results/retrain_summary.csv"); s = s[s.trigger == "every_slice"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for rule, c, name in [("strict", C2, "strict (all facets improve)"), ("pareto", C1, "non-inferior + one better")]:
        a = s[(s.rule == rule) & (s.scenario == "combined_prior")].sort_values("delta_mult")
        b = s[(s.rule == rule) & (s.scenario == "none")].sort_values("delta_mult")
        axes[0].plot(a.delta_mult, a.d_eu_pct, color=c, marker="o", ms=5, mec="white", label=name)
        axes[1].plot(b.delta_mult, b.promo_per_run, color=c, marker="o", ms=5, mec="white", label=name)
    axes[0].set_ylabel("utility gain vs. no retraining, %"); axes[0].set_title("(a) harmful gradual drift", fontsize=8.5, loc="left", color=INK)
    axes[1].set_ylabel("promotions per run"); axes[1].set_title("(b) no drift: needless switches", fontsize=8.5, loc="left", color=INK)
    for ax in axes:
        ax.set_xlabel("tolerance multiplier (× base δ)"); ax.set_xticks([0, 0.5, 1, 2, 4])
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    save(fig, "fig6_promotion")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
