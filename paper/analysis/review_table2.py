"""R7(c): Table 2 re-run from data/records.csv with every row written as parameters
(test statistic, multiplicity procedure, delta allocation). GLM, eps = 5%, delta = 0.1, 200 50/50
splits, violation against the full-pool risk. Writes data/review_table2.json."""
import json, os, sys, numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import load, outcome_matrix, p_betting, SETTINGS, HERE
from review_replay import splits, sub
EPS, DELTA = 0.05, 0.1

def pvals(L, test, delta):
    n = L.shape[1]; H = (L > 0).astype(float)
    if test == 'betting': return p_betting(L, EPS, delta)
    if test == 'binom': return binom.cdf(H.sum(1), n, EPS)
    if test in ('hoef_h', 'hoef_l'):
        v = H if test == 'hoef_h' else L; m = v.mean(1); rng = 1 if test == 'hoef_h' else 2
        return np.where(m >= EPS, 1.0, np.exp(-2 * n * (EPS - m) ** 2 / rng ** 2))

ROWS = [  # name, tests (union if two, delta split equally), procedure
    ('Betting, signed loss (ours)', ['betting'], 'fixed'),
    ('Betting $\\cup$ binomial', ['betting', 'binom'], 'fixed'),
    ('Binomial, harm', ['binom'], 'fixed'),
    ('Betting, Bonferroni', ['betting'], 'bonf'),
    ('Binomial, Bonferroni', ['binom'], 'bonf'),
    ('Hoeffding, harm', ['hoef_h'], 'fixed'),
    ('Hoeffding, signed loss', ['hoef_l'], 'fixed')]

def select(L, C, tests, proc):
    M = L.shape[0] - 1; dl = DELTA / len(tests) / (M if proc == 'bonf' else 1)
    ok = np.zeros(M, bool)
    for t in tests: ok |= pvals(L[1:], t, dl) <= dl
    if proc == 'fixed':
        f = np.nonzero(~ok)[0]; ok[:] = False; ok[:f[0] if len(f) else M] = True
    cand = [0] + [i + 1 for i in np.nonzero(ok)[0]]
    return min(cand, key=lambda j: C[j].mean())

if __name__ == '__main__':
    d = load(); out = {}
    for m, b in [s for s in SETTINGS if s[0] == 'GLM']:
        x = sub(d, m, b); Y, C, L = outcome_matrix(x); R = L.mean(1); CK = x.CK.to_numpy(float); o = {}
        for name, tests, proc in ROWS:
            tok, v = [], 0
            for cal, ev in splits(len(x)):
                j = select(L[:, cal], C[:, cal], tests, proc); tok.append(C[j, ev].mean() / CK[ev].mean()); v += R[j] > EPS
            o[name] = [100 * float(np.mean(tok)), 100 * v / 200]
        out[b] = o
    json.dump(dict(rows=[r[0] for r in ROWS], res=out), open(os.path.join(HERE, '..', 'data', 'review_table2.json'), 'w'), indent=1)
    for name, *_ in ROWS: print(f'{name:<32}', ' '.join(f"{out[b][name][0]:5.0f}%/{out[b][name][1]:.1f}" for b in out))
