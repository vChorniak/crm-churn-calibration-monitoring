"""
Retraining + composite promotion rule experiment (uses the same worlds, models and slices
as simulation.py: identical seeds and random streams).

Calendar convention: slice t is scored at time t; its labels arrive at time t+1.
  label-free trigger fires at time t   -> candidate trained on slice t-1 (latest labelled) -> serves t+1 ...
  label-based trigger for slice t fires at t+1 -> candidate trained on slice t -> serves t+2 ...
Candidate = same pipeline (L1-LR + isotonic + Mondrian conformal), fitted on a 60/20/20 split of the
labelled slice (train / calibration / validation). The composite rule (Section "profile") compares the
candidate with the incumbent on the validation part; promotion only if all four conditions hold.
Run: python retraining.py -> results/retrain.csv
"""
import sys, csv, warnings
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import average_precision_score
from simulation import (World, Model, drift_path, lift, murphy, N, N_STABLE, N_SLICES, ALPHA,
                      BENEFIT, COST, TOP_Q, SEEDS, SCENARIOS, psi)

warnings.filterwarnings("ignore")
SCEN = ["none", "combined_prior", "concept", "abrupt"]
DELTA = dict(rank=0.005, cal=0.001, lift=0.05, cov=0.02)
MULTS = [0.0, 0.5, 1.0, 2.0, 4.0]
TRIGGERS = ["never", "u_mond", "psi_010", "brier", "reliability", "every_slice"]


def profile(model, X, y):
    p = model.predict(X)
    i0, i1 = model.sets(p, mondrian=True)
    return dict(pr=average_precision_score(y, p), brier=float(np.mean((p - y) ** 2)),
                lift=lift(y, p), cov=float(np.where(y == 1, i1, i0).mean()))


def promote(cand, inc, m, rule="strict"):
    """strict = rule as formulated in the article draft (all four facets must improve / hold);
    pareto = non-inferior on every facet within delta AND better by more than delta on at least one."""
    if rule == "pareto":
        d = m
        noninf = (cand["pr"] >= inc["pr"] - d * DELTA["rank"] and cand["brier"] <= inc["brier"] + d * DELTA["cal"]
                  and cand["lift"] >= inc["lift"] - d * DELTA["lift"] and cand["cov"] >= 1 - ALPHA - d * DELTA["cov"])
        sup = (cand["pr"] > inc["pr"] + d * DELTA["rank"] or cand["brier"] < inc["brier"] - d * DELTA["cal"]
               or cand["lift"] > inc["lift"] + d * DELTA["lift"])
        return noninf and sup
    return (cand["pr"] >= inc["pr"] + m * DELTA["rank"] and cand["brier"] <= inc["brier"] + m * DELTA["cal"]
            and cand["lift"] >= inc["lift"] + m * DELTA["lift"] and cand["cov"] >= 1 - ALPHA - m * DELTA["cov"])


def eu(y, p):
    k = int(len(y) * TOP_Q); top = np.argsort(-p)[:k]
    return float((y[top] * BENEFIT - COST).sum() / 1000)


def fit_candidate(X, y, rng):
    idx = rng.permutation(len(y)); a, b = int(0.6 * len(y)), int(0.8 * len(y))
    tr, ca, va = idx[:a], idx[a:b], idx[b:]
    return Model(X[tr], y[tr], X[ca], y[ca]), (X[va], y[va])


def run_seed(seed):
    rng = np.random.default_rng(seed)
    world = World(rng)
    X_tr, y_tr = world.sample(N, rng); X_cal, y_cal = world.sample(N, rng)
    base = Model(X_tr, y_tr, X_cal, y_cal)
    ref_p = base.predict(X_cal)
    out = []
    for scen in SCEN:
        srng = np.random.default_rng([seed, SCENARIOS.index(scen)])
        slices = [world.sample(N, srng, *drift_path(scen, t), prior_fixed=(scen == "combined_prior")) for t in range(1, N_SLICES + 1)]
        # signals of the incumbent on every slice
        sig = {"u_mond": [], "psi": [], "brier": [], "rel": []}
        for X, y in slices:
            p = base.predict(X); i0, i1 = base.sets(p, True)
            sig["u_mond"].append(((i0.astype(int) + i1) != 1).mean()); sig["psi"].append(psi(ref_p, p))
            sig["brier"].append(np.mean((p - y) ** 2)); sig["rel"].append(murphy(y, p)[0])
        lim = {k: (np.mean(v[:N_STABLE]), np.std(v[:N_STABLE], ddof=1)) for k, v in sig.items()}

        def first_alarm(trig):
            for t in range(N_STABLE + 1, N_SLICES + 1):
                i = t - 1
                if trig == "u_mond" and abs(sig["u_mond"][i] - lim["u_mond"][0]) > 3 * lim["u_mond"][1]:
                    return t, t - 1          # (alarm time, labelled slice used for retraining)
                if trig == "psi_010" and sig["psi"][i] > 0.10:
                    return t, t - 1
                if trig == "brier" and sig["brier"][i] > lim["brier"][0] + 3 * lim["brier"][1] and t < N_SLICES:
                    return t + 1, t
                if trig == "reliability" and sig["rel"][i] > lim["rel"][0] + 3 * lim["rel"][1] and t < N_SLICES:
                    return t + 1, t
            return None

        cache = {}

        def cand_for(t_lab):
            if t_lab not in cache:
                cache[t_lab] = fit_candidate(*slices[t_lab - 1], np.random.default_rng([seed, t_lab, SCENARIOS.index(scen)]))
            return cache[t_lab]
        for rule, trig in [(ru, tr) for ru in ("strict", "pareto") for tr in TRIGGERS]:
            if trig == "never" and rule == "pareto":
                continue
            for m in (MULTS if trig != "never" else [1.0]):
                current, promoted_at, n_prom, n_cand = base, None, 0, 0
                serve = {t: base for t in range(1, N_SLICES + 1)}
                if trig == "every_slice":        # retrain at every drift-period time step, always via the rule
                    for t in range(N_STABLE + 1, N_SLICES):
                        cand, (Xv, yv) = cand_for(t - 1)
                        n_cand += 1
                        if promote(profile(cand, Xv, yv), profile(current, Xv, yv), m, rule):
                            current = cand; n_prom += 1
                        for s in range(t + 1, N_SLICES + 1):
                            serve[s] = current
                elif trig != "never":
                    fa = first_alarm(trig)
                    if fa and fa[0] < N_SLICES:
                        t_alarm, t_lab = fa
                        cand, (Xv, yv) = cand_for(t_lab)
                        n_cand = 1
                        if promote(profile(cand, Xv, yv), profile(base, Xv, yv), m, rule):
                            n_prom = 1; promoted_at = t_alarm + 1
                            for s in range(t_alarm + 1, N_SLICES + 1):
                                serve[s] = cand
                drift_ts = range(N_STABLE + 1, N_SLICES + 1)
                ev = [(eu(y, serve[t].predict(X)), average_precision_score(y, serve[t].predict(X)),
                       float(np.mean((serve[t].predict(X) - y) ** 2))) for t, (X, y) in zip(drift_ts, slices[N_STABLE:])]
                ev = np.array(ev)
                out.append(dict(seed=seed, scenario=scen, rule=rule, trigger=trig, delta_mult=m, n_candidates=n_cand,
                                n_promotions=n_prom, promoted_at=promoted_at if promoted_at else -1,
                                eu_mean=ev[:, 0].mean(), eu_last=ev[-1, 0], pr_mean=ev[:, 1].mean(),
                                brier_mean=ev[:, 2].mean()))
    return out


if __name__ == "__main__":
    seeds = SEEDS if len(sys.argv) < 2 else SEEDS[: int(sys.argv[1])]
    with Pool(2) as pool:
        rows = [r for o in pool.map(run_seed, seeds) for r in o]
    with open("results/retrain.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("done", len(rows))
