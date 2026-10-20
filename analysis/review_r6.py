"""R6: escalation policies A-E (section 6.3) as training and calibration sizes vary separately.
Configurations (train/calibration, evaluation = rest): 20/20, 40/20, 60/20 and 40/10, 40/30.
200 splits each (GLM, eps = 5%, delta = 0.1). Per split and method: tokens on the evaluation part
(% of reference), violation against the full-pool risk, the selected parameter (direct = run the
reference). Paired differences to A are per split. Writes paper/data/review_r6.json."""
import json, os, random, statistics as st, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, 'analysis')
from climb_calib import calibrate, GRIDX
from escalation_variants import pol_A, mk_B, pol_C, mk_D, best_offset, TH, LAMS
from opt_escalation import load_family, make_policy, run_policy
EPS, DELTA = 0.05, 0.1; BEN = ['MATH', 'GAIA', 'FRAMES', 'MuSiQue']
CONFIGS = [(0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.4, 0.1), (0.4, 0.3)]; REPS, CHUNK = 200, 25

def work(args):
    (ftr, fca), chunk = args; rows = load_family('GLM'); random.seed(1000 * chunk + int(100 * ftr) + int(10 * fca))
    pools = {b: [x for x in rows if x['ds'] == b] for b in BEN}; res = []
    for _ in range(CHUNK):
        tr, ca, te = [], [], []
        for b in BEN:
            g = pools[b][:]; random.shuffle(g); n = len(g)
            tr += g[:int(ftr * n)]; ca += g[int(ftr * n):int((ftr + fca) * n)]; te += g[int((ftr + fca) * n):]
        fD = mk_D(tr); predE = make_policy(tr); rec = {}
        for b in BEN:
            trb, cab, teb = ([x for x in s if x['ds'] == b] for s in (tr, ca, te))
            fB = mk_B(best_offset(trb, EPS)); fE = lambda x, lam, S=None: run_policy([x], predE([x]), lam)[0]
            rb = {}
            for m, f, grid in (('A', pol_A, GRIDX), ('B', fB, TH + [-1e9]), ('C', pol_C, TH + [-1e9]), ('D', fD, LAMS), ('E', fE, LAMS)):
                lam = calibrate(cab, EPS, DELTA, grid=grid, policy=f)
                r = [f(x, lam) for x in teb]; rp = [f(x, lam) for x in pools[b]]
                rb[m] = dict(tok=st.mean(c for _, c in r) / st.mean(x['cP'] for x in teb),
                             viol=st.mean(int(x['okP']) - int(ok) for x, (ok, _) in zip(pools[b], rp)) > EPS,
                             lam=str(lam))
            rb['n'] = [len(trb), len(cab), len(teb)]; rec[b] = rb
        res.append(rec)
    return f'{ftr}/{fca}', res

if __name__ == '__main__':
    jobs = [(c, k) for c in CONFIGS for k in range(REPS // CHUNK)]
    with Pool(20) as p: out = p.map(work, jobs)
    raw = {}
    for k, r in out: raw.setdefault(k, []).extend(r)
    rng = np.random.default_rng(0); summ = {}
    for k, reps in raw.items():
        summ[k] = {}
        for b in BEN:
            s = dict(n=reps[0][b]['n'])
            A = np.array([r[b]['A']['tok'] for r in reps])
            for m in 'ABCDE':
                t = np.array([r[b][m]['tok'] for r in reps]); dlt = t - A
                bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
                lams = [r[b][m]['lam'] for r in reps]
                s[m] = dict(tok=100 * t.mean(), viol=100 * np.mean([r[b][m]['viol'] for r in reps]),
                            diff=100 * dlt.mean(), diff_ci=[100 * np.quantile(bs, .025), 100 * np.quantile(bs, .975)],
                            direct=100 * np.mean([l == 'direct' for l in lams]),
                            top=sorted(((lams.count(l), l) for l in set(lams)), reverse=True)[:3])
            summ[k][b] = s
    json.dump(dict(summary=summ), open('paper/data/review_r6.json', 'w'), indent=1)
    for k, v in summ.items():
        print(k, ' | '.join(f"{b}: " + ' '.join(f"{m}{v[b][m]['tok']:.0f}" for m in 'ABCDE') for b in BEN))
