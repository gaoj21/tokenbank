"""Separate the two sources of certified savings: the loss (signed vs. clipped at zero) and the test
(betting vs. Hoeffding-Bentkus vs. Hoeffding, plus the exact binomial for the 0/1 clipped loss).
Every cell uses the same fixed-sequence procedure, candidate family, splits and delta.

Signed loss l = Y_ref - Y_pi in {-1, 0, 1} is mapped affinely to z = (l + 1) / 2 in [0, 1] with the
tolerance mapped to (eps + 1) / 2, which keeps the negative values and the control target (the
accuracy loss). The clipped loss max(l, 0) is the harm indicator; controlling its mean at eps
controls the harm rate, a different and larger quantity. Hoeffding-Bentkus follows Learn-then-Test
(Angelopoulos et al., 2021): p = min(exp(-n h(min(r, a), a)), e * P(Bin(n, a) <= ceil(n r))) for a
[0,1]-bounded loss with empirical mean r and level a.
Deployed procedure: the ladder depth is chosen on calibration (climb_families in single_decision.py).
usage: python analysis/loss_test_factorial.py [records.csv] -> data/loss_test_factorial.json"""
import json, math, os, sys, numpy as np, pandas as pd
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import p_betting, HERE
from single_decision import climb_families

EPS = (0.02, 0.05, 0.10); DELTA = 0.1; SPLITS = 200

def kl(a, b):
    a = np.clip(a, 1e-12, 1 - 1e-12); b = min(max(b, 1e-12), 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

def p_hb(Z, a):
    """Hoeffding-Bentkus p-values for rows of Z in [0,1], null mean > a."""
    n = Z.shape[1]; r = Z.mean(1)
    hoef = np.where(r < a, np.exp(-n * kl(np.minimum(r, a), a)), 1.0)
    bent = math.e * binom.cdf(np.ceil(n * r), n, a)
    return np.minimum(np.minimum(hoef, bent), 1.0)

def p_hoeffding(Z, a):
    n = Z.shape[1]; r = Z.mean(1)
    return np.where(r < a, np.exp(-2 * n * (a - r) ** 2), 1.0)

def p_binom(H, a):
    return binom.cdf(H.sum(1), H.shape[1], a)

def pvals(L, loss, test, eps, delta=DELTA):
    if loss == 'signed':
        if test == 'betting':
            return p_betting(L, eps, delta)
        Z, a = (L + 1) / 2, (eps + 1) / 2
    else:
        Z, a = np.maximum(L, 0).astype(float), eps
        if test == 'betting':
            return p_betting(Z, eps, delta)
        if test == 'binomial':
            return p_binom(Z, a)
    return p_hb(Z, a) if test == 'hb' else p_hoeffding(Z, a)

def select(L, C, loss, test, eps, delta=DELTA):
    p = pvals(L[1:], loss, test, eps, delta); fail = np.nonzero(p > delta)[0]
    m = fail[0] if len(fail) else len(p)
    return int(np.argmin(C[:m + 1].mean(1)))

def select_multi(fams, cal, loss, test, eps):
    """As deployed: each depth family is tested at delta / 2, the cheaper one on calibration is kept."""
    best = None
    for f, (Y, C, L, YK, CK) in enumerate(fams):
        j = select(L[:, cal], C[:, cal], loss, test, eps, DELTA / len(fams)); c = C[j, cal].mean()
        if best is None or c < best[0]: best = (c, f, j)
    return best[1], best[2]

CELLS = [('signed', 'betting'), ('signed', 'hb'), ('signed', 'hoeffding'),
         ('clipped', 'betting'), ('clipped', 'hb'), ('clipped', 'hoeffding'), ('clipped', 'binomial')]

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'records_new.csv')
    d = pd.read_csv(path); out = {}
    for (m, b), x in d.groupby(['model', 'bench'], sort=False):
        x = x.reset_index(drop=True); fams = climb_families(x); CK = fams[0][4]; rng = np.random.default_rng(0)
        harm_rate = 100 * float(((x.Y0 == 1) & (x.YK == 0)).mean())       # team breaks a correct L0 answer
        o = dict(n=len(x), harm_rate=harm_rate)
        for e in EPS:
            res = {f'{l}|{t}': [[], 0] for l, t in CELLS}
            for _ in range(SPLITS):
                pm = rng.permutation(len(x)); cal, ev = pm[:len(x) // 2], pm[len(x) // 2:]
                for l, t in CELLS:
                    f, j = select_multi(fams, cal, l, t, e); C, L = fams[f][1], fams[f][2]; r = res[f'{l}|{t}']
                    r[0].append(C[j, ev].mean() / CK[ev].mean()); r[1] += L[j].mean() > e
            o[str(e)] = {k: dict(tok=100 * float(np.mean(v[0])), viol=100 * v[1] / SPLITS) for k, v in res.items()}
        out[f'{m}|{b}'] = o
        print(f"{m:10s} {b:8s} harm {harm_rate:4.1f}% | " + ' | '.join(
            f"e{100*e:.0f}: " + ' '.join(f"{o[str(e)][f'{l}|{t}']['tok']:.0f}" for l, t in CELLS) for e in EPS))
    print('columns:', ', '.join(f'{l}/{t}' for l, t in CELLS))
    json.dump(out, open(os.path.join(HERE, '..', 'data', 'loss_test_factorial.json'), 'w'), indent=1)
