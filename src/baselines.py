"""
Baselines and ablation.
(1) Synthetic, 20 seeds: stable-period quality and detection delays for
    LR-L1 + isotonic (proposed), LR-L1 without recalibration, unweighted LR-L1 + isotonic,
    histogram gradient boosting + isotonic, recency heuristic (quality only).
(2) Real data (rolling origin as in real_data.py): LR-L1 + isotonic vs gradient boosting + isotonic.
Run: python baselines.py -> results/baselines_synth.csv, results/baselines_synth_detect.csv, results/baselines_real.csv
"""
import warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from simulation import World, drift_path, murphy, lift, N, N_STABLE, N_SLICES, SCENARIOS, ALPHA, BENEFIT, COST, TOP_Q
from detection_analysis import detect

warnings.filterwarnings("ignore")
SEEDS = list(range(1000, 1020))
SCEN = ["none", "combined_prior", "concept"]


class Pipe:
    def __init__(self, kind, X_tr, y_tr, X_cal, y_cal):
        self.kind = kind
        w = np.where(y_tr == 1, (1 - y_tr.mean()) / y_tr.mean(), 1.0)
        self.sc = StandardScaler().fit(X_tr)
        if kind == "gbm":
            self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                      early_stopping=True, random_state=0).fit(X_tr, y_tr, sample_weight=w)
        else:
            self.clf = LogisticRegression(penalty="l1", C=0.05, solver="liblinear", max_iter=3000, random_state=0).fit(
                self.sc.transform(X_tr), y_tr, sample_weight=None if kind == "lr_unweighted" else w)
        self.iso = None if kind == "lr_no_recal" else IsotonicRegression(out_of_bounds="clip").fit(self._raw(X_cal), y_cal)

    def _raw(self, X):
        if self.kind == "gbm":
            return self.clf.predict_proba(X)[:, 1]
        return self.clf.decision_function(self.sc.transform(X))

    def predict(self, X):
        r = self._raw(X)
        if self.iso is None:
            r = 1 / (1 + np.exp(-r))
        else:
            r = self.iso.predict(r)
        return np.clip(r, 1e-6, 1 - 1e-6)


def qual(y, p):
    rel, res, ece = murphy(y, p)
    k = int(len(y) * TOP_Q); top = np.argsort(-p)[:k]
    return dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p), brier=np.mean((p - y) ** 2),
                reliability=rel, ece=ece, lift=lift(y, p), eu_decile=(y[top] * BENEFIT - COST).sum() / 1000,
                mean_p=p.mean())


KINDS = ["lr_l1_iso", "lr_no_recal", "lr_unweighted", "gbm"]


def run_seed(seed):
    rng = np.random.default_rng(seed)
    world = World(rng)
    X_tr, y_tr = world.sample(N, rng); X_cal, y_cal = world.sample(N, rng)
    pipes = {k: Pipe(k, X_tr, y_tr, X_cal, y_cal) for k in KINDS}
    rows = []
    for scen in SCEN:
        srng = np.random.default_rng([seed, SCENARIOS.index(scen)])
        for t in range(1, N_SLICES + 1):
            X, y = world.sample(N, srng, *drift_path(scen, t), prior_fixed=(scen == "combined_prior"))
            for k, pp in pipes.items():
                rows.append(dict(seed=seed, scenario=scen, slice=t, model=k, **qual(y, pp.predict(X))))
            rows.append(dict(seed=seed, scenario=scen, slice=t, model="heuristic_recency", **qual(y, (X[:, 0] - X[:, 0].min()) / np.ptp(X[:, 0]) * 0.999 + 1e-4)))
    return rows


def real_part():
    from real_data import kkbox, retail
    rows = []
    for ds, origins in [(kkbox(), None), (retail(), list(range(0, 12)))]:
        name, X, y, key, months, cols, lag = ds
        if origins is None:
            origins = [months.index(m) for m in ["201601", "201602", "201603", "201604", "201605", "201606", "201607", "201608"]]
        for o in origins:
            idx = np.random.default_rng(o).permutation(np.where(key == o)[0])
            a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
            tr, ca, te = idx[:a], idx[a:b], idx[b:]
            for k in ["lr_l1_iso", "gbm"]:
                pp = Pipe(k, X[tr], y[tr], X[ca], y[ca])
                for h in range(0, 7):
                    if o + h >= len(months):
                        break
                    sel = te if h == 0 else np.where(key == o + h)[0]
                    q = qual(y[sel], pp.predict(X[sel])); q.pop("eu_decile")
                    rows.append(dict(dataset=name, origin=months[o], horizon=h, model=k, **q))
            print(name, months[o], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with Pool(2) as pool:
        rows = [r for o in pool.map(run_seed, SEEDS) for r in o]
    d = pd.DataFrame(rows); d.to_csv("results/baselines_synth.csv", index=False)
    det = []
    for (seed, scen, m), g in d.groupby(["seed", "scenario", "model"]):
        g = g.sort_values("slice")
        for sig, direc in [("brier", 1), ("reliability", 1), ("ece", 1), ("pr_auc", -1), ("roc_auc", -1), ("lift", -1), ("mean_p", 0)]:
            det.append(dict(seed=seed, scenario=scen, model=m, signal=sig, delay=detect(g[sig].values, direc, "3sigma")))
    pd.DataFrame(det).to_csv("results/baselines_synth_detect.csv", index=False)
    real_part().to_csv("results/baselines_real.csv", index=False)
    print("done")
