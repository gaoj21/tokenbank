"""Rebuttal baseline: per-setting empirical tuning without certification. On each calibration half, deploy the
cheapest candidate (either family) whose empirical mean loss is at most eps, and compare with CLIMB on the same
splits as Table 1 (main_results.replay: seed 0, 200 splits, violation = full-pool loss above eps).
usage: python analysis/plugin_baseline.py  ->  data/plugin_baseline.json"""
import json, os, numpy as np
from main_results import families, SPLITS
from single_decision import calibrate_multi, EPS
from final_analyses import frame, SET
from climb_core import HERE

def plugin(F, cal, e):
    best = None
    for f, (Y, C, L, YK, CK) in enumerate(F):
        r = L[:, cal].mean(1); c = C[:, cal].mean(1); ok = np.where(r <= e)[0]
        j = ok[np.argmin(c[ok])]; 
        if best is None or c[j] < best[0]: best = (c[j], f, j)
    return best[1], best[2]

out = {}
for m, b in SET:
    x = frame(m, b); F = families(x, x.ag.values); n = len(x); CK = F[0][4]; rng = np.random.default_rng(0); r = {}
    for e in EPS:
        s = {k: dict(acc=[], tok=[], loss=[], v=0) for k in ('climb', 'plugin')}
        for _ in range(SPLITS):
            p = rng.permutation(n); cal, ev = p[:n // 2], p[n // 2:]
            for k, fn in (('climb', calibrate_multi), ('plugin', plugin)):
                f, j = fn(F, cal, e); Y, C, L, _, _ = F[f]
                s[k]['acc'].append(Y[j, ev].mean()); s[k]['tok'].append(C[j, ev].mean() / CK[ev].mean())
                s[k]['loss'].append(L[j].mean()); s[k]['v'] += L[j].mean() > e
        r[str(e)] = {k: dict(acc=100 * float(np.mean(v['acc'])), tok=100 * float(np.mean(v['tok'])),
                             loss=100 * float(np.mean(v['loss'])), worst_loss=100 * float(np.max(v['loss'])),
                             viol=100 * v['v'] / SPLITS) for k, v in s.items()}
    out[f'{m}|{b}'] = r
    print(m, b, {e: (round(r[e]['climb']['viol'], 1), round(r[e]['plugin']['viol'], 1), round(r[e]['climb']['tok']), round(r[e]['plugin']['tok'])) for e in r}, flush=True)
json.dump(out, open(os.path.join(HERE, '..', 'data', 'plugin_baseline.json'), 'w'), indent=1)
