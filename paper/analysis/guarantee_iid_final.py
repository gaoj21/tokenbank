"""Direct check of Proposition 2 for the deployed procedure (main_results.py) on the final records. The task
distribution is the recorded pool; calibration sets of size N/2 are drawn i.i.d. from it, CLIMB is calibrated
(two families at delta/2 each), and the chosen policy's exact risk on the pool is computed. Writes
data/guarantee_iid_final.json."""
import json, os, sys, numpy as np
from multiprocessing import Pool
from scipy.stats import beta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import HERE
from single_decision import calibrate_multi
from final_analyses import frame, SET
from main_results import families

REPS = int(os.environ.get('REPS', 5000)); EPS = (0.02, 0.05, 0.10)

def cp(k, n, a=0.05):
    return (0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1), 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k))

def run(s):
    m, b = s; x = frame(m, b); F = families(x, x.ag.values); N = len(x); n = N // 2
    rng = np.random.default_rng(SET.index(s)); out = {}
    for e in EPS:
        v, tok, risk = 0, [], []
        for _ in range(REPS):
            idx = rng.integers(0, N, n); f, j = calibrate_multi(F, idx, e)
            Y, C, L, _, CK = F[f]; R = L[j].mean()
            v += R > e; tok.append(C[j].mean() / CK.mean()); risk.append(R)
        lo, hi = cp(v, REPS)
        out[str(e)] = dict(viol=v / REPS, ci=[lo, hi], tok=float(np.mean(tok)), risk=float(np.mean(risk)), max_risk=float(np.max(risk)))
    return f'{m}|{b}', dict(N=N, n=n, reps=REPS, out=out)

if __name__ == '__main__':
    with Pool(12) as p:
        res = dict(p.map(run, SET))
    json.dump(res, open(os.path.join(HERE, '..', 'data', 'guarantee_iid_final.json'), 'w'), indent=1)
    for k, r in res.items():
        print(k, ' '.join(f"eps={e}: viol {100*o['viol']:.2f}% [{100*o['ci'][0]:.2f},{100*o['ci'][1]:.2f}]" for e, o in r['out'].items()))
