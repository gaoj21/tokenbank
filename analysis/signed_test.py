"""Tighter calibration: test the signed loss directly with a variance-adaptive betting test.
H_m: E[l] > eps, l = Y_pipeline - Y_policy in {-1, 0, 1}. Under H_m the capital
K_t = prod_i (1 + lam_i (eps - l_i)) is a nonnegative supermartingale for predictable
lam_i in [0, 1/(1-eps)), so p = 1 / max_t K_t is a valid p-value (Ville). lam_i uses only
earlier samples (Waudby-Smith & Ramdas style). Compared with the paper's test (exact binomial on
the harm indicator) on the same splits, grid extended to -inf (always stop at the cheap rungs)."""
import json, math, random, statistics as st, sys
from scipy.stats import binom
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items, apply, GRID
EPS, DELTA, REPS = 0.05, 0.1, 200
GRIDX = GRID + [round(-0.6 - 0.05 * i, 2) for i in range(1, 49)] + [-1e9]   # down to -3.0, then -inf

def p_binom_h(losses_h, n): return binom.cdf(sum(losses_h), n, EPS)

def p_betting(l):
    n = len(l); K, Kmax = 1.0, 1.0; s1, s2 = 0.0, 0.0
    cap = 0.5 / (1 - EPS)
    for t, x in enumerate(l, 1):
        mu = (s1 + EPS) / t; var = (s2 + 0.25) / t - 0.0      # smoothed running moments of earlier samples
        var = max((s2 + 0.25) / t - ((s1) / t) ** 2, 1e-3)
        lam = min(cap, math.sqrt(2 * math.log(1 / DELTA) / (var * n)))
        K *= 1 + lam * (EPS - x); Kmax = max(Kmax, K)
        s1 += x; s2 += x * x
    return 1.0 / Kmax

def select(cal, grid, kind):
    n = len(cal); cert = ['direct']
    for lam in grid[1:]:
        r = [apply(x, lam) for x in cal]
        if kind == 'binom':
            p = p_binom_h([int(x['okP'] and not ok) for x, (ok, c) in zip(cal, r)], n)
        else:
            p = p_betting([int(x['okP']) - int(ok) for x, (ok, c) in zip(cal, r)])
        if p <= DELTA: cert.append(lam)
        else: break
    return min(cert, key=lambda lam: st.mean(apply(x, lam)[1] for x in cal))

if __name__ == '__main__':
    out = {}
    random.seed(4)
    for ds, ae, nf in CELLS['GLM']:
        it = items(ae, nf)
        res = {k: {'tok': [], 'loss': [], 'viol': 0} for k in ('binom', 'betting')}
        for _ in range(REPS):
            random.shuffle(it); h = len(it) // 2; cal, te = it[:h], it[h:]
            cte = st.mean(x['cP'] for x in te)
            for k, grid in (('binom', GRID), ('betting', GRIDX)):
                lam = select(cal, grid, k); r = [apply(x, lam) for x in te]
                loss = st.mean(int(x['okP']) - int(ok) for x, (ok, c) in zip(te, r))
                res[k]['tok'].append(st.mean(c for ok, c in r) / cte); res[k]['loss'].append(loss); res[k]['viol'] += loss > EPS
        allL1 = (100 * st.mean(x['ok1'] for x in it), 100 * st.mean(x['c1'] for x in it) / st.mean(x['cP'] for x in it))
        aP = 100 * st.mean(x['okP'] for x in it)
        out[ds] = {k: dict(tok=round(100 * st.mean(v['tok'])), acc=round(aP - 100 * st.mean(v['loss']), 1),
                           viol=round(100 * v['viol'] / REPS, 1)) for k, v in res.items()}
        out[ds]['pipeline_acc'] = round(aP, 1); out[ds]['always_L1'] = [round(allL1[0], 1), round(allL1[1])]
        print(ds, out[ds], flush=True)
    json.dump(out, open('results/signed_test_glm.json', 'w'), indent=1)
