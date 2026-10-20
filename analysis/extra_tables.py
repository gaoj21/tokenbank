"""Two extra result tables, computed offline from the replay records.
(1) Baselines on the full test set: always-L0, always-L1, fixed threshold, oracle router,
    CLIMB (eps = 5%, mean over 200 calibration/evaluation splits) and the full pipeline.
(2) Calibration ablation at eps = 5%, delta = 0.1: our fixed-sequence exact binomial test on the
    harm indicator vs. Bonferroni over the grid, Hoeffding on the harm indicator, and Hoeffding on
    the signed loss. Reports tokens (% of pipeline) and violation rate over 200 splits."""
import json, math, random, statistics as st, sys
from scipy.stats import binom
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items, apply, GRID, top
EPS, DELTA, REPS = 0.05, 0.1, 200

def oracle(x):
    if x['ok0']: return True, x['c0']
    if x['ok1']: return True, x['c1']
    return top(x)

def p_value(cal, lam, kind):
    n = len(cal); res = [apply(x, lam) for x in cal]
    h = [int(x['okP'] and not ok) for x, (ok, c) in zip(cal, res)]
    if kind in ('binom', 'bonf'):
        return binom.cdf(sum(h), n, EPS)
    if kind == 'hoef_h':
        m = st.mean(h); return 1.0 if m >= EPS else math.exp(-2 * n * (EPS - m) ** 2)
    l = [int(x['okP']) - int(ok) for x, (ok, c) in zip(cal, res)]          # signed loss in [-1, 1]
    m = st.mean(l); return 1.0 if m >= EPS else math.exp(-n * (EPS - m) ** 2 / 2)

def select(cal, kind):
    cert = ['direct']
    if kind == 'bonf':
        cert += [lam for lam in GRID[1:] if p_value(cal, lam, kind) <= DELTA / (len(GRID) - 1)]
    else:
        for lam in GRID[1:]:
            if p_value(cal, lam, kind) <= DELTA: cert.append(lam)
            else: break
    return min(cert, key=lambda lam: st.mean(apply(x, lam)[1] for x in cal))

base, abl = {}, {}
random.seed(3)
for fam, cells in CELLS.items():
    for ds, ae, nf in cells:
        it = items(ae, nf); cP = st.mean(x['cP'] for x in it); aP = 100 * st.mean(x['okP'] for x in it)
        def summ(f): r = [f(x) for x in it]; return (round(100 * st.mean(ok for ok, c in r), 1), round(100 * st.mean(c for ok, c in r) / cP))
        base[f'{fam}|{ds}'] = {'L0': summ(lambda x: (x['ok0'], x['c0'])), 'L1': summ(lambda x: (x['ok1'], x['c1'])),
                               'fixed': summ(lambda x: apply(x, -0.10)), 'oracle': summ(oracle), 'pipeline': (round(aP, 1), 100)}
        acc = {k: [] for k in ('binom', 'bonf', 'hoef_h', 'hoef_l')}; viol = {k: 0 for k in acc}
        for _ in range(REPS):
            random.shuffle(it); h = len(it) // 2; cal, te = it[:h], it[h:]
            cte = st.mean(x['cP'] for x in te)
            for k in acc:
                lam = select(cal, k); r = [apply(x, lam) for x in te]
                acc[k].append(st.mean(c for ok, c in r) / cte)
                viol[k] += st.mean(int(x['okP']) - int(ok) for x, (ok, c) in zip(te, r)) > EPS
        abl[f'{fam}|{ds}'] = {k: (round(100 * st.mean(acc[k])), round(100 * viol[k] / REPS, 1)) for k in acc}
        print(fam, ds, base[f'{fam}|{ds}'], abl[f'{fam}|{ds}'], flush=True)
json.dump({'baselines': base, 'calibration_ablation': abl}, open('results/extra_tables.json', 'w'), indent=1)
