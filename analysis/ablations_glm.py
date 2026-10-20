"""GLM-only ablations with the final calibration (analysis/climb_calib.py), eps = 5%, delta = 0.1.
(1) Escalation policies under the same guarantee: A shared threshold (CLIMB), B two thresholds,
    C agreement stop, D Bayes deciles, E learned values. 40/20/40 train/calibration/evaluation.
(2) Calibration design: combined test (CLIMB) vs. binomial-on-harm only, betting-on-loss only,
    Bonferroni over the grid, Hoeffding on harm, Hoeffding on signed loss. 50/50 splits.
200 repetitions each; violations against the risk on the full pool."""
import json, math, random, statistics as st, sys
import numpy as np
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items, apply, GRID
from climb_calib import calibrate, GRIDX, p_binom, p_betting
from escalation_variants import pol_A, mk_B, pol_C, mk_D, best_offset, TH, LAMS
from opt_escalation import load_family, make_policy, run_policy, BENCH
import os
EPS, DELTA = 0.05, 0.1
REPS = int(os.environ.get('REPS', '200')); SEED = int(os.environ.get('SEED', '0'))
PART = os.environ.get('PART', 'all'); OUT = os.environ.get('OUT', 'results/ablations_glm.json')
BEN = ['MATH', 'GAIA', 'FRAMES', 'MuSiQue']

def evaluate(te, f, lam, pool):
    r = [f(x, lam) for x in te]
    return (st.mean(c for ok, c in r) / st.mean(x['cP'] for x in te),
            st.mean(int(x['okP']) - int(ok) for x, (ok, c) in zip(te, r)))

# ---------- (1) policies ----------
pol, polv = {}, {}
if PART in ('all', 'policies'):
    rows = load_family('GLM')
    pol = {b: {m: [] for m in 'ABCDE'} for b in BEN}; polv = {b: {m: 0 for m in 'ABCDE'} for b in BEN}
    random.seed(11 + SEED)
    for rep in range(REPS):
        tr, ca, te = [], [], []
        for b in BEN:
            g = [x for x in rows if x['ds'] == b]; random.shuffle(g); n = len(g)
            tr += g[:int(.4 * n)]; ca += g[int(.4 * n):int(.6 * n)]; te += g[int(.6 * n):]
        fD = mk_D(tr); predE = make_policy(tr)
        for b in BEN:
            trb = [x for x in tr if x['ds'] == b]; cab = [x for x in ca if x['ds'] == b]; teb = [x for x in te if x['ds'] == b]
            fB = mk_B(best_offset(trb, EPS))
            fE = lambda x, lam, S=None: run_policy([x], predE([x]), lam)[0]
            for m, f, grid in (('A', pol_A, GRIDX), ('B', fB, TH + [-1e9]), ('C', pol_C, TH + [-1e9]), ('D', fD, LAMS), ('E', fE, LAMS)):
                lam = calibrate(cab, EPS, DELTA, grid=grid, policy=f)
                tok, loss = evaluate(teb, f, lam, None)
                pol[b][m].append(tok); polv[b][m] += loss > EPS
        if rep % 20 == 0: print('policies rep', rep, flush=True)
    policies = {b: {m: round(100 * st.mean(v), 1) for m, v in pol[b].items()} for b in BEN}
    print('policies', policies, flush=True)

# ---------- (2) calibration design ----------
def select(cal, kind):
    cert = ['direct']
    if kind == 'bonf':
        for lam in GRIDX[1:]:
            r = [apply(x, lam) for x in cal]; h = [int(x['okP'] and not ok) for x, (ok, c) in zip(cal, r)]
            if p_binom(h, EPS) <= DELTA / (len(GRIDX) - 1): cert.append(lam)
        return min(cert, key=lambda lam: st.mean(apply(x, lam)[1] for x in cal))
    if kind in ('combined', 'binom', 'betting'):
        return calibrate(cal, EPS, DELTA, grid=GRIDX, test=kind)
    for lam in GRIDX[1:]:
        r = [apply(x, lam) for x in cal]; n = len(cal)
        v = [int(x['okP'] and not ok) for x, (ok, c) in zip(cal, r)] if kind == 'hoef_h' else [int(x['okP']) - int(ok) for x, (ok, c) in zip(cal, r)]
        m = st.mean(v); rng = 1 if kind == 'hoef_h' else 2
        p = 1.0 if m >= EPS else math.exp(-2 * n * (EPS - m) ** 2 / rng ** 2)
        if p <= DELTA: cert.append(lam)
        else: break
    return min(cert, key=lambda lam: st.mean(apply(x, lam)[1] for x in cal))
KINDS = ['combined', 'binom', 'betting', 'bonf', 'hoef_h', 'hoef_l']
calib, calib_raw = {}, {}
random.seed(12 + SEED)
for ds, ae, nf in (CELLS['GLM'] if PART in ('all', 'calib') else []):
    it = items(ae, nf); pool = {}
    def prisk(lam):
        if lam not in pool: pool[lam] = st.mean(int(x['okP']) - int(apply(x, lam)[0]) for x in it)
        return pool[lam]
    acc = {k: [] for k in KINDS}; vio = {k: 0 for k in KINDS}
    for _ in range(REPS):
        random.shuffle(it); h = len(it) // 2; cal, te = it[:h], it[h:]
        for k in KINDS:
            lam = select(cal, k); tok, loss = evaluate(te, apply, lam, None)
            acc[k].append(tok); vio[k] += prisk(lam) > EPS
    calib[ds] = {k: [round(100 * st.mean(acc[k])), round(100 * vio[k] / REPS, 1)] for k in KINDS}
    calib_raw[ds] = {k: dict(tok=acc[k], viol=vio[k]) for k in KINDS}
    print('calib', ds, calib[ds], flush=True)
json.dump({'policies_raw': pol, 'policy_viol_counts': polv, 'calib_raw': calib_raw, 'reps': REPS}, open(OUT, 'w'))
