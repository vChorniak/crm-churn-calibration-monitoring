"""
Label-noise robustness of the base learner: L1 (two strengths) vs L2 vs unregularised logistic regression.
Training and calibration labels are flipped at rate r (symmetric noise); evaluation is on a clean
stationary slice. Also records how many correlated-noise and pure-noise features receive a zero weight.
Run: python label_noise.py -> results/noise.csv
"""
import sys, csv, warnings
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import average_precision_score, roc_auc_score
from simulation import World, Model, murphy, lift, N

warnings.filterwarnings("ignore")
RATES = [0.0, 0.05, 0.10, 0.20]
LEARNERS = {"L1_C0.05": ("l1", 0.05), "L1_C0.005": ("l1", 0.005), "L1_C0.001": ("l1", 0.001),
            "L2_C0.05": ("l2", 0.05), "none": (None, None)}
SEEDS = list(range(2000, 2030))


def flip(y, r, rng):
    m = rng.random(len(y)) < r
    return np.where(m, 1 - y, y)


def run_seed(seed):
    rng = np.random.default_rng(seed)
    world = World(rng)
    X_tr, y_tr = world.sample(N, rng); X_cal, y_cal = world.sample(N, rng); X_te, y_te = world.sample(N, rng)
    rows = []
    for r in RATES:
        nrng = np.random.default_rng([seed, int(r * 100)])
        ytr, ycal = flip(y_tr, r, nrng), flip(y_cal, r, nrng)
        for name, (pen, C) in LEARNERS.items():
            m = Model(X_tr, ytr, X_cal, ycal, C=C if C else 1.0, penalty=pen)
            p = m.predict(X_te); rel, res, ece = murphy(y_te, p); coef = m.clf.coef_[0]
            rows.append(dict(seed=seed, noise=r, learner=name, pr_auc=average_precision_score(y_te, p),
                             roc_auc=roc_auc_score(y_te, p), brier=float(np.mean((p - y_te) ** 2)), ece=ece,
                             lift=lift(y_te, p), zero_core=int((coef[:7] == 0).sum()),
                             zero_corr_noise=int((coef[7:12] == 0).sum()), zero_pure_noise=int((coef[12:] == 0).sum()),
                             l1_norm_noise=float(np.abs(coef[7:]).sum()), l1_norm_core=float(np.abs(coef[:7]).sum())))
    return rows


if __name__ == "__main__":
    seeds = SEEDS if len(sys.argv) < 2 else SEEDS[: int(sys.argv[1])]
    with Pool(2) as pool:
        rows = [r for o in pool.map(run_seed, seeds) for r in o]
    with open("results/noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("done", len(rows))
