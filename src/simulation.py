"""
Synthetic CRM drift experiment, version 2 (Q1 revision).

Changes vs. v1 (experiment_code.py):
  * clean generator: standardisation moments and the intercept are fixed on a stationary
    reference sample, so "covariate-only" drift is a pure covariate shift (P(Y|X) unchanged)
    and "concept-only" drift changes P(Y|X) only; the base rate is allowed to move;
  * the mixing matrix of the correlated noise features is fixed per seed; five independent
    pure-noise features are added (tests L1 selection);
  * five scenarios: none (false alarms), combined ramp, covariate ramp, concept ramp, abrupt;
  * 12 stable reference slices (control limits from 12 points), 6 drift slices;
  * class-conditional (Mondrian) split-conformal layer in addition to the marginal one;
  * calibration-specific metrics: ECE, Murphy reliability/resolution, calibration slope;
  * label-free signals: mean score, expected Brier (confidence-based), PSI and KS on scores,
    max feature PSI, conformal ambiguity (marginal and Mondrian);
  * identical alarm rules for every signal (3-sigma, and 2-of-3 beyond 2-sigma), plus
    industry PSI thresholds 0.10 / 0.25;
  * 50 independent seeds.
Run: python simulation.py  ->  results/slices.csv, results/model_info.csv
"""
import os, sys, json, warnings
import numpy as np
from multiprocessing import Pool
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

N = 50_000
N_STABLE = 12
N_DRIFT = 6
N_SLICES = N_STABLE + N_DRIFT
STEP = 0.11
BASE_RATE = 0.06
ALPHA = 0.10
BENEFIT, COST = 1000.0, 100.0
TOP_Q = 0.10
SEEDS = list(range(1000, 1050))
SCENARIOS = ["none", "combined", "covariate", "concept", "abrupt", "combined_prior", "misspec_none", "misspec_covariate"]
MAIN_SCEN, EXTRA_SCEN = SCENARIOS[:5], ["combined_prior"]   # misspec_* scenarios tested, model stayed calibrated -> not reported
GAMMA = -1.0  # concave (saturating) recency effect in the misspecified world
N_CORE = 7


def drift_path(scen, t):
    """(covariate intensity, concept intensity) at slice t (1-based)."""
    k = t - N_STABLE
    if k <= 0 or scen == "none":
        return 0.0, 0.0
    ramp = STEP * k
    if scen == "misspec_none":
        return 0.0, 0.0
    return {"combined": (ramp, ramp), "covariate": (ramp, 0.0), "concept": (0.0, ramp),
            "abrupt": (STEP * N_DRIFT, STEP * N_DRIFT), "combined_prior": (ramp, ramp),
            "misspec_covariate": (ramp, 0.0)}[scen]


class World:
    """Generative process with seed-fixed nuisance structure and reference moments."""
    BETA = np.array([1.15, -0.95, -0.55, -0.35, 0.25, 0.12, 0.10])
    CONCEPT = np.array([0.45, -0.55, 0, 0, 0, 0, 0])

    def __init__(self, rng, misspec=False):
        self.misspec = misspec
        self.mix = rng.normal(0, 0.5, (3, 5))
        ref = self._core(200_000, rng, 0.0)
        self.mu, self.sd = ref.mean(0), ref.std(0)
        lin = self._lin(ref, self.BETA) + rng.normal(0, 0.9, len(ref))
        from scipy.optimize import brentq
        self.c = brentq(lambda c: (1 / (1 + np.exp(-(lin - c)))).mean() - BASE_RATE, -20, 20)

    @staticmethod
    def _core(n, rng, dc):
        recency = rng.exponential(30 * (1 + 0.35 * dc), n)
        frequency = rng.poisson(max(0.5, 6 * (1 - 0.30 * dc)), n).astype(float)
        monetary = rng.lognormal(4.2 - 0.25 * dc, 0.8, n)
        tenure = rng.gamma(2.0, 180, n)
        contacts = rng.poisson(3, n).astype(float)
        lr, lf, lm = np.log1p(recency), np.log1p(frequency), np.log1p(monetary)
        return np.column_stack([lr, lf, lm, np.log1p(tenure), np.log1p(contacts), lr * lf, lm / (1 + lf)])

    def _lin(self, core, beta):
        z = (core - self.mu) / self.sd
        lin = z @ beta
        if self.misspec:            # convex recency effect that a linear model cannot represent
            lin = lin + GAMMA * np.maximum(0.0, z[:, 0] - 0.5) ** 2
        return lin

    def sample(self, n, rng, dc=0.0, dk=0.0, prior_fixed=False):
        core = self._core(n, rng, dc)
        corr_noise = core[:, :3] @ self.mix + rng.normal(0, 1, (n, 5))
        pure_noise = rng.normal(0, 1, (n, 5))
        X = np.hstack([core, corr_noise, pure_noise])
        beta = self.BETA * (1 + dk * self.CONCEPT)
        lin = self._lin(core, beta) + rng.normal(0, 0.9, n)
        c = self.c
        if prior_fixed:             # v1-style: intercept re-solved so that the base rate stays at BASE_RATE
            from scipy.optimize import brentq
            c = brentq(lambda c: (1 / (1 + np.exp(-(lin - c)))).mean() - BASE_RATE, -20, 20)
        y = (rng.random(n) < 1 / (1 + np.exp(-(lin - c)))).astype(int)
        return X, y


class Model:
    """Eq. (4): class-weighted L1 logistic regression + isotonic recalibration + conformal layers."""

    def __init__(self, X_tr, y_tr, X_cal, y_cal, C=0.05, penalty="l1"):
        self.sc = StandardScaler().fit(X_tr)
        w = np.where(y_tr == 1, (1 - y_tr.mean()) / y_tr.mean(), 1.0)
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        kw = dict(penalty=penalty, C=C, solver=solver, max_iter=3000, random_state=0) if penalty else dict(C=1e6, max_iter=3000, random_state=0)
        self.clf = LogisticRegression(**kw).fit(self.sc.transform(X_tr), y_tr, sample_weight=w)
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(self._raw(X_cal), y_cal)
        p = self.predict(X_cal)
        s = np.where(y_cal == 1, 1 - p, p)          # s(x, y) = 1 - p(y|x)
        self.q_marg = self._q(s)
        self.q_mond = {c: self._q(s[y_cal == c]) for c in (0, 1)}

    @staticmethod
    def _q(s):
        n = len(s)
        return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")

    def _raw(self, X):
        return self.clf.decision_function(self.sc.transform(X))

    def predict(self, X):
        return np.clip(self.iso.predict(self._raw(X)), 1e-6, 1 - 1e-6)

    def sets(self, p, mondrian):
        q0, q1 = (self.q_mond[0], self.q_mond[1]) if mondrian else (self.q_marg, self.q_marg)
        return p <= q0, (1 - p) <= q1          # include 0, include 1


# ------------------------------------------------------------------ metrics
def lift(y, p, q=TOP_Q):
    k = int(len(y) * q)
    return y[np.argsort(-p)[:k]].mean() / max(y.mean(), 1e-9)


def psi(ref, cur, bins=10):
    e_ = np.quantile(ref, np.linspace(0, 1, bins + 1)); e_[0], e_[-1] = -np.inf, np.inf
    e_ = np.unique(e_)
    e = np.clip(np.histogram(ref, e_)[0] / len(ref), 1e-6, None)
    a = np.clip(np.histogram(cur, e_)[0] / len(cur), 1e-6, None)
    return float(((a - e) * np.log(a / e)).sum())


def murphy(y, p, bins=20):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ybar = y.mean(); rel = res = ece = 0.0
    for b in np.unique(idx):
        m = idx == b; w = m.mean(); pk, yk = p[m].mean(), y[m].mean()
        rel += w * (pk - yk) ** 2; res += w * (yk - ybar) ** 2; ece += w * abs(pk - yk)
    return rel, res, ece


def cal_slope(y, p):
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=500).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def slice_metrics(model, X, y, p, ref_p, ref_X):
    i0m, i1m = model.sets(p, mondrian=False)
    i0c, i1c = model.sets(p, mondrian=True)
    size_m, size_c = i0m.astype(int) + i1m, i0c.astype(int) + i1c
    rel, res, ece = murphy(y, p)
    slope, icpt = cal_slope(y, p)
    k = int(len(y) * TOP_Q)
    top = np.argsort(-p)[:k]
    thr = np.where(p > COST / BENEFIT)[0]
    return dict(
        base=float(y.mean()),
        # label-based
        brier=float(np.mean((p - y) ** 2)), reliability=rel, resolution=res, ece=ece,
        cal_slope=slope, cal_icpt=icpt,
        roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
        lift=float(lift(y, p)), accuracy=float(((p >= 0.5) == y).mean()),
        cov_marg=float(np.where(y == 1, i1m, i0m).mean()),
        cov_marg_pos=float(i1m[y == 1].mean()), cov_marg_neg=float(i0m[y == 0].mean()),
        cov_mond=float(np.where(y == 1, i1c, i0c).mean()),
        cov_mond_pos=float(i1c[y == 1].mean()), cov_mond_neg=float(i0c[y == 0].mean()),
        eu_decile=float((y[top] * BENEFIT - COST).sum() / 1000),
        eu_thresh=float((y[thr] * BENEFIT - COST).sum() / 1000), n_thresh=int(len(thr)),
        # label-free
        mean_p=float(p.mean()), exp_brier=float(np.mean(p * (1 - p))),
        psi_score=psi(ref_p, p), ks_score=float(ks_2samp(ref_p, p).statistic),
        psi_feat_max=max(psi(ref_X[:, j], X[:, j]) for j in range(N_CORE)),
        u_marg=float((size_m != 1).mean()), u_marg_empty=float((size_m == 0).mean()),
        u_mond=float((size_c != 1).mean()), u_mond_both=float((size_c == 2).mean()),
        u_mond_empty=float((size_c == 0).mean()),
        share_p_gt_qmarg=float((p > model.q_marg).mean()),
    )


def run_seed(seed):
    rng = np.random.default_rng(seed)
    world = World(rng)
    X_tr, y_tr = world.sample(N, rng)
    X_cal, y_cal = world.sample(N, rng)
    model = Model(X_tr, y_tr, X_cal, y_cal)
    ref_p, ref_X = model.predict(X_cal), X_cal
    coef = model.clf.coef_[0]
    info = dict(seed=seed, q_marg=float(model.q_marg), q_mond0=float(model.q_mond[0]),
                q_mond1=float(model.q_mond[1]), zero_core=int((coef[:7] == 0).sum()),
                zero_corr_noise=int((coef[7:12] == 0).sum()), zero_pure_noise=int((coef[12:] == 0).sum()))
    rows = []
    for scen in MAIN_SCEN:
        srng = np.random.default_rng([seed, SCENARIOS.index(scen)])
        for t in range(1, N_SLICES + 1):
            dc, dk = drift_path(scen, t)
            X, y = world.sample(N, srng, dc, dk)
            p = model.predict(X)
            r = dict(seed=seed, scenario=scen, slice=t, dc=dc, dk=dk)
            r.update(slice_metrics(model, X, y, p, ref_p, ref_X))
            rows.append(r)
    return info, rows


def run_seed_extra(seed):
    rows = []
    rng = np.random.default_rng(seed)
    world = World(rng)
    X_tr, y_tr = world.sample(N, rng); X_cal, y_cal = world.sample(N, rng)
    model = Model(X_tr, y_tr, X_cal, y_cal)
    if any(sc.startswith("misspec") for sc in EXTRA_SCEN):
        mrng = np.random.default_rng([seed, 777])
        mworld = World(mrng, misspec=True)
        Xm_tr, ym_tr = mworld.sample(N, mrng); Xm_cal, ym_cal = mworld.sample(N, mrng)
        mmodel = Model(Xm_tr, ym_tr, Xm_cal, ym_cal)
    for scen in EXTRA_SCEN:
        w, m, Xc = (world, model, X_cal) if scen == "combined_prior" else (mworld, mmodel, Xm_cal)  # noqa
        ref_p = m.predict(Xc)
        srng = np.random.default_rng([seed, SCENARIOS.index(scen)])
        for t in range(1, N_SLICES + 1):
            dc, dk = drift_path(scen, t)
            X, y = w.sample(N, srng, dc, dk, prior_fixed=(scen == "combined_prior"))
            p = m.predict(X)
            r = dict(seed=seed, scenario=scen, slice=t, dc=dc, dk=dk)
            r.update(slice_metrics(m, X, y, p, ref_p, Xc))
            rows.append(r)
    return rows


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "extra":
    import csv
    with Pool(2) as pool:
        rows = [r for o in pool.map(run_seed_extra, SEEDS) for r in o]
    with open("results/slices_extra.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("done extra", len(rows))
elif __name__ == "__main__":
    import csv
    os.makedirs("results", exist_ok=True)
    seeds = SEEDS if len(sys.argv) < 2 else SEEDS[: int(sys.argv[1])]
    with Pool(2) as pool:
        out = pool.map(run_seed, seeds)
    infos = [o[0] for o in out]; rows = [r for o in out for r in o[1]]
    for name, data in [("model_info", infos), ("slices", rows)]:
        with open(f"results/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    print("done", len(rows))
