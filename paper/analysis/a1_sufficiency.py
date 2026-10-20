"""Empirical check of (A1) sufficiency. A1 is only restrictive at L1, the one stopping point with an
earlier score: given S1, the later outcomes and cost must not depend on S0. For each later quantity
we compare a cross-fitted model that sees only the current state with one that also sees the
history, and report the held-out improvement with a task-level bootstrap CI.
  targets: Y1 (answer if stopping at L1), Y2 (seeded pipeline), YK (reference pipeline; enters the
           loss), log(C2 - C1) (incremental cost of escalating from L1)
  current: S1 (mean log-prob); history adds S0.   Variant 'withH': current S1,H1; history adds S0,H0.
Binary targets: logistic regression on cubic B-spline features; cost: ridge on the same features.
5-fold cross-fitting repeated 10 times, per-task held-out predictions averaged over repeats;
2,000 bootstrap resamples of tasks. Writes paper/data/a1_sufficiency.json."""
import json, os, sys, numpy as np, warnings
from multiprocessing import Pool
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import load, SETTINGS, HERE
warnings.filterwarnings('ignore')
REPEATS, FOLDS, BOOT = 10, 5, 2000
TARGETS = ['Y1', 'Y2', 'YK', 'logdC']
VARIANTS = {'S': (['S1'], ['S1', 'S0']), 'withH': (['S1', 'H1'], ['S1', 'H1', 'S0', 'H0'])}

def model(binary):
    sp = SplineTransformer(n_knots=5, degree=3, extrapolation='constant')
    return make_pipeline(StandardScaler(), sp, LogisticRegression(C=1.0, max_iter=2000) if binary else Ridge(alpha=1.0))

def crossfit(X, y, binary):
    P = np.zeros(len(y))
    for r in range(REPEATS):
        kf = (StratifiedKFold if binary else KFold)(FOLDS, shuffle=True, random_state=r)
        for tr, te in kf.split(X, y):
            m = model(binary).fit(X[tr], y[tr])
            P[te] += (m.predict_proba(X[te])[:, 1] if binary else m.predict(X[te])) / REPEATS
    return P

def run(args):
    (mname, b), var = args
    d = load(); d = d[(d.model == mname) & (d.bench == b)].reset_index(drop=True)
    d['logdC'] = np.log1p(np.maximum(d.C2 - d.C1, 0))
    cur, his = VARIANTS[var]; rng = np.random.default_rng(0); out = {}
    for t in TARGETS:
        y = d[t].to_numpy(float); binary = t != 'logdC'
        if binary and (y.min() == y.max() or min(y.sum(), len(y) - y.sum()) < FOLDS):
            out[t] = None; continue
        Pc = crossfit(d[cur].to_numpy(float), y, binary); Ph = crossfit(d[his].to_numpy(float), y, binary)
        if binary:
            Pc, Ph = np.clip(Pc, 1e-4, 1 - 1e-4), np.clip(Ph, 1e-4, 1 - 1e-4)
            lc = -(y * np.log(Pc) + (1 - y) * np.log(1 - Pc)); lh = -(y * np.log(Ph) + (1 - y) * np.log(1 - Ph))
        else:
            lc, lh = (y - Pc) ** 2, (y - Ph) ** 2
        D = lc - lh                                                      # > 0: history helps
        bs = np.array([D[i].mean() for i in (rng.integers(0, len(D), len(D)) for _ in range(BOOT))])
        r = dict(base=float(lc.mean()), gain=float(D.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                 rel=float(D.mean() / lc.mean()))
        if binary:
            ac, ah = roc_auc_score(y, Pc), roc_auc_score(y, Ph); ba = []
            for _ in range(BOOT):
                i = rng.integers(0, len(y), len(y))
                if y[i].min() != y[i].max(): ba.append(roc_auc_score(y[i], Ph[i]) - roc_auc_score(y[i], Pc[i]))
            r.update(auc_cur=ac, auc_his=ah, dauc=ah - ac, dauc_ci=[float(np.quantile(ba, .025)), float(np.quantile(ba, .975))])
        out[t] = r
    return f'{mname}|{b}|{var}', dict(n=len(d), out=out)

if __name__ == '__main__':
    with Pool(16) as p: res = dict(p.map(run, [(s, v) for v in VARIANTS for s in SETTINGS]))
    json.dump(res, open(os.path.join(HERE, '..', 'data', 'a1_sufficiency.json'), 'w'), indent=1)
    for k, r in res.items():
        print(k, r['n'], ' | '.join(f"{t}: {1000*o['gain']:+.1f} [{1000*o['ci'][0]:+.1f},{1000*o['ci'][1]:+.1f}] ({100*o['rel']:+.1f}%)" + (f" dAUC {o['dauc']:+.3f} [{o['dauc_ci'][0]:+.3f},{o['dauc_ci'][1]:+.3f}]" if 'dauc' in o else '')
                                     if o else f'{t}: n/a' for t, o in r['out'].items()))
