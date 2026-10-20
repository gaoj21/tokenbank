"""Tolerance sweep for the final method: tokens (% of pipeline), accuracy and violation rate as
eps varies, one setting per process (argv: fam ds). 100 calibration/evaluation splits per eps."""
import json, random, statistics as st, sys
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items, apply
from climb_calib import calibrate
fam, ds = sys.argv[1], sys.argv[2]
EPS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10]; REPS = 100
ae, nf = [(a, n) for d, a, n in CELLS[fam] if d == ds][0]
it = items(ae, nf); aP = 100 * st.mean(x['okP'] for x in it); pool = {}
def prisk(lam):
    if lam not in pool: pool[lam] = st.mean(int(x['okP']) - int(apply(x, lam)[0]) for x in it)
    return pool[lam]
random.seed(31); out = {}
for eps in EPS:
    tok, acc, v = [], [], 0
    for _ in range(REPS):
        random.shuffle(it); cal, te = it[:len(it) // 2], it[len(it) // 2:]
        lam = calibrate(cal, eps); r = [apply(x, lam) for x in te]
        tok.append(100 * st.mean(c for ok, c in r) / st.mean(x['cP'] for x in te)); acc.append(100 * st.mean(ok for ok, c in r))
        v += prisk(lam) > eps
    out[str(eps)] = dict(tok=round(st.mean(tok), 1), acc=round(st.mean(acc), 1), viol=round(100 * v / REPS, 1))
    print(fam, ds, eps, out[str(eps)], flush=True)
json.dump(dict(fam=fam, ds=ds, pipe_acc=round(aP, 1), sweep=out), open(f'results/eps_sweep_{fam}_{ds}.json', 'w'), indent=1)
