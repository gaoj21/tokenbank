"""Direct check of Proposition 2. The task distribution is taken to be the finite recorded pool
(uniform over its tasks); calibration sets of size n are drawn i.i.d. from it (with replacement),
CLIMB is calibrated, and the deployed policy's risk under that distribution is computed exactly.
The failure event R(pi_hat) > eps is exactly the one Proposition 2 bounds by delta, with no
evaluation-set noise. Writes paper/data/guarantee_iid.json."""
import json, sys, os, numpy as np
from multiprocessing import Pool
from scipy.stats import beta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import load, outcome_matrix, calibrate, SETTINGS, HERE

REPS = int(os.environ.get('REPS', 5000)); EPS = (0.02, 0.05, 0.10); DELTA = 0.1

def cp(k, n, a=0.05):
    return (0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1), 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k))

def run(args):
    m, b = args
    d = load(); d = d[(d.model == m) & (d.bench == b)].reset_index(drop=True)
    Y, C, L = outcome_matrix(d); N = len(d); n = N // 2
    R = L.mean(1); Cm = C.mean(1)                                      # exact risk / cost on the pool
    rng = np.random.default_rng(SETTINGS.index((m, b))); out = {}
    for eps in EPS:
        v, tok, risk = 0, [], []
        for _ in range(REPS):
            idx = rng.integers(0, N, n)
            j = calibrate(L[:, idx], C[:, idx], eps, DELTA)
            v += R[j] > eps; tok.append(Cm[j] / Cm[0]); risk.append(R[j])
        lo, hi = cp(v, REPS)
        out[str(eps)] = dict(viol=v / REPS, ci=[lo, hi], tok=float(np.mean(tok)), risk=float(np.mean(risk)), max_risk=float(np.max(risk)))
    return f'{m}|{b}', dict(N=N, n=n, reps=REPS, out=out)

if __name__ == '__main__':
    with Pool(8) as p: res = dict(p.map(run, SETTINGS))
    json.dump(res, open(os.path.join(HERE, '..', 'data', 'guarantee_iid.json'), 'w'), indent=1)
    for k, r in res.items():
        print(k, r['N'], ' '.join(f"eps={e}: viol {100*o['viol']:.2f}% [{100*o['ci'][0]:.2f},{100*o['ci'][1]:.2f}] tok {100*o['tok']:.0f}% risk {100*o['risk']:.2f}"
                                for e, o in r['out'].items()))
