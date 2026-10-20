"""GLM only. Same as signed_test.py, plus (i) violation measured against the risk on the full pool
(the finite population the splits are drawn from), in addition to the evaluation-half estimate, and
(ii) a combined test that rejects if either the binomial-on-harm test or the betting test rejects at
level delta/2 (valid by a union bound)."""
import json, random, statistics as st, sys
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items, apply, GRID
from signed_test import p_binom_h, p_betting, GRIDX, EPS, DELTA
REPS = 200

def pvals(cal, lam):
    r = [apply(x, lam) for x in cal]; n = len(cal)
    pb = p_binom_h([int(x['okP'] and not ok) for x, (ok, c) in zip(cal, r)], n)
    pw = p_betting([int(x['okP']) - int(ok) for x, (ok, c) in zip(cal, r)])
    return pb, pw

def select(cal, grid, kind):
    cert = ['direct']
    for lam in grid[1:]:
        pb, pw = pvals(cal, lam)
        ok = {'binom': pb <= DELTA, 'betting': pw <= DELTA, 'combined': min(pb, pw) <= DELTA / 2}[kind]
        if ok: cert.append(lam)
        else: break
    return min(cert, key=lambda lam: st.mean(apply(x, lam)[1] for x in cal))

out = {}
random.seed(5)
for ds, ae, nf in CELLS['GLM']:
    it = items(ae, nf)
    pool_risk = {}
    def prisk(lam):
        if lam not in pool_risk:
            pool_risk[lam] = st.mean(int(x['okP']) - int(apply(x, lam)[0]) for x in it)
        return pool_risk[lam]
    res = {k: {'tok': [], 'loss': [], 'v_eval': 0, 'v_pool': 0} for k in ('binom', 'betting', 'combined')}
    for _ in range(REPS):
        random.shuffle(it); h = len(it) // 2; cal, te = it[:h], it[h:]
        cte = st.mean(x['cP'] for x in te)
        for k in res:
            grid = GRID if k == 'binom' else GRIDX
            lam = select(cal, grid, k); r = [apply(x, lam) for x in te]
            loss = st.mean(int(x['okP']) - int(ok) for x, (ok, c) in zip(te, r))
            res[k]['tok'].append(st.mean(c for ok, c in r) / cte); res[k]['loss'].append(loss)
            res[k]['v_eval'] += loss > EPS; res[k]['v_pool'] += prisk(lam) > EPS
    aP = 100 * st.mean(x['okP'] for x in it)
    out[ds] = {k: dict(tok=round(100 * st.mean(v['tok'])), acc=round(aP - 100 * st.mean(v['loss']), 1),
                       viol_eval=round(100 * v['v_eval'] / REPS, 1), viol_pool=round(100 * v['v_pool'] / REPS, 1)) for k, v in res.items()}
    out[ds]['pipeline_acc'] = round(aP, 1)
    print(ds, out[ds], flush=True)
json.dump(out, open('results/signed_test_glm2.json', 'w'), indent=1)
