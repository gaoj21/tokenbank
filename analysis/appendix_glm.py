"""GLM appendix analyses with the final calibration (betting on the signed loss), eps = 5%.
(1) Sensitivity to delta in {0.05, 0.1, 0.2} and to calibration size n in {50, 100, 200, all-half}.
(2) Distribution shift: calibrate on one benchmark, deploy on another (all ordered pairs);
    report tokens and the true risk on the target pool vs eps."""
import json, os, random, statistics as st, sys
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items, apply
from climb_calib import calibrate
EPS, REPS = 0.05, int(os.environ.get('REPS', '100'))
cells = {ds: items(ae, nf) for ds, ae, nf in CELLS['GLM']}
def risk(pool, lam): return st.mean(int(x['okP']) - int(apply(x, lam)[0]) for x in pool)
def tok(pool, lam): return st.mean(apply(x, lam)[1] for x in pool) / st.mean(x['cP'] for x in pool)
out = {'delta': {}, 'n': {}, 'shift': {}}
random.seed(21)
for ds, it in cells.items():
    pool = {}
    for delta in (0.05, 0.1, 0.2):
        t, v = [], 0
        for _ in range(REPS):
            random.shuffle(it); cal, te = it[:len(it) // 2], it[len(it) // 2:]
            lam = calibrate(cal, EPS, delta); t.append(tok(te, lam)); v += risk(it, lam) > EPS
        out['delta'].setdefault(ds, {})[str(delta)] = [round(100 * st.mean(t)), round(100 * v / REPS, 1)]
    for n in (50, 100, 200, len(it) // 2):
        if n > len(it) // 2: continue
        t, v = [], 0
        for _ in range(REPS):
            random.shuffle(it); cal, te = it[:n], it[len(it) // 2:]
            lam = calibrate(cal, EPS); t.append(tok(te, lam)); v += risk(it, lam) > EPS
        out['n'].setdefault(ds, {})[str(n)] = [round(100 * st.mean(t)), round(100 * v / REPS, 1)]
    print(ds, out['delta'][ds], out['n'][ds], flush=True)
for src, a in cells.items():
    lam = calibrate(a, EPS)
    for dst, b in cells.items():
        if src == dst: continue
        out['shift'].setdefault(src, {})[dst] = dict(threshold=lam, tok=round(100 * tok(b, lam)), risk=round(100 * risk(b, lam), 1))
    print('calibrated on', src, '->', out['shift'][src], flush=True)
json.dump(out, open('results/appendix_glm.json', 'w'), indent=1)
