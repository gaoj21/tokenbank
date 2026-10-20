"""Replay analyses requested in review (R1-R5, R7), all on data/records.csv with the conventions of
climb_core: loss = YK - Y_pi against the reference pipeline (no draft); a rung stops if S_k > t and
H_k <= 0.6; escalation takes (Y2, C2); the empty action runs the reference (YK, CK). Intervals over
random 50/50 splits reflect split randomness only, not independent re-runs of the benchmarks.
Usage: python analysis/review_replay.py r1|r2|r3|r4|r5|r7|all   -> data/review_<item>.json"""
import json, os, sys, numpy as np
from multiprocessing import Pool
from scipy.stats import binomtest, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import load, outcome_matrix, calibrate, SETTINGS, GRID, TAU, HERE
DATA = os.path.join(HERE, '..', 'data'); EPS = (0.02, 0.05, 0.10); DELTA = 0.1; SPLITS = 200
sub = lambda d, m, b: d[(d.model == m) & (d.bench == b)].reset_index(drop=True)
dump = lambda name, obj: json.dump(obj, open(os.path.join(DATA, f'review_{name}.json'), 'w'), indent=1)

def splits(N, reps=SPLITS, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        p = rng.permutation(N); yield p[:N // 2], p[N // 2:]

def evaluate(Y, C, L, YK, CK, eps, reps=SPLITS):
    """Calibrate on one half, evaluate on the other; candidates are rows (row 0 = reference)."""
    tok, loss, viol, picked, R = [], [], 0, [], L.mean(1)
    for cal, ev in splits(L.shape[1], reps):
        j = calibrate(L[:, cal], C[:, cal], eps, DELTA)
        tok.append(C[j, ev].mean() / CK[ev].mean()); loss.append(YK[ev].mean() - Y[j, ev].mean())
        viol += R[j] > eps; picked.append(j)
    return dict(tok=float(np.mean(tok)), loss=float(np.mean(loss)), viol=viol / reps,
                certified=float(np.mean(np.array(picked) > 0)), picked=np.bincount(picked, minlength=L.shape[0]).tolist())

def rows(d, Ys, Cs):
    YK, CK = d.YK.to_numpy(int), d.CK.to_numpy(float)
    Y = np.array([YK] + Ys, int); C = np.array([CK] + Cs, float)
    return Y, C, YK[None, :] - Y, YK, CK

# ---------------- R1: reference pipeline vs. seeded L2 ----------------
def r1(d):
    out = {}
    for m, b in SETTINGS:
        x = sub(d, m, b); b01 = int(((x.YK == 1) & (x.Y2 == 0)).sum()); b10 = int(((x.YK == 0) & (x.Y2 == 1)).sum())
        out[f'{m}|{b}'] = dict(n=len(x), accK=100 * x.YK.mean(), acc2=100 * x.Y2.mean(), diff=100 * (x.Y2 - x.YK).mean(),
                               p=binomtest(b10, b01 + b10).pvalue if b01 + b10 else 1.0, disagree=100 * (x.YK != x.Y2).mean(),
                               CK=x.CK.mean(), C2=x.C2.mean(), team=(x.C2 - x.C1).mean(), C1=x.C1.mean())
    return out

# ---------------- R2: randomised mixtures and untied thresholds ----------------
def mixtures(x, qs=(0.25, 0.5, 0.75), seed=0):
    """Fixed sequence t_1, mix(t_1,t_2;q)..., t_2, ... ; per-task lottery u_i fixed by seed."""
    Y, C, _ = outcome_matrix(x); u = np.random.default_rng(seed).random(len(x)); Ys, Cs = [], []
    for m in range(1, Y.shape[0]):
        if m > 1:
            for q in qs:
                agg = u < q; Ys.append(np.where(agg, Y[m], Y[m - 1])); Cs.append(np.where(agg, C[m], C[m - 1]))
        Ys.append(Y[m]); Cs.append(C[m])
    return rows(x, Ys, Cs)

def ladder(x, t0, t1, top=('Y2', 'C2'), rungs=(0, 1)):
    S0, H0, S1, H1 = (x[k].to_numpy(float) for k in ('S0', 'H0', 'S1', 'H1'))
    s0 = (S0 > t0) & (H0 <= TAU) if 0 in rungs else np.zeros(len(x), bool)
    s1 = ~s0 & (S1 > t1) & (H1 <= TAU) if 1 in rungs else np.zeros(len(x), bool)
    Ytop = x[top[0]].to_numpy(); Ctop = x[top[1]].to_numpy(float) + (x.C0.to_numpy(float) if top == ('YK', 'CK') else 0)
    return np.where(s0, x.Y0, np.where(s1, x.Y1, Ytop)), np.where(s0, x.C0, np.where(s1, x.C1, Ctop))

OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10)
def untied_eval(x, eps, reps=SPLITS):
    """t1 = t0 + offset; one fixed sequence per offset at level delta/5 (Bonferroni); cheapest certified."""
    seqs = []
    for o in OFFSETS:
        YY, CC = zip(*(ladder(x, t, t + o) for t in GRID)); seqs.append(rows(x, list(YY), list(CC)))
    YK, CK = seqs[0][3], seqs[0][4]; tok, loss, viol, cert = [], [], 0, 0
    for cal, ev in splits(len(x), reps):
        best = (CK[cal].mean(), None)
        for Y, C, L, _, _ in seqs:
            j = calibrate(L[:, cal], C[:, cal], eps, DELTA / len(OFFSETS))
            if j and C[j, cal].mean() < best[0]: best = (C[j, cal].mean(), (Y, C, L, j))
        if best[1] is None: tok.append(1.0); loss.append(0.0); continue
        Y, C, L, j = best[1]; cert += 1
        tok.append(C[j, ev].mean() / CK[ev].mean()); loss.append(YK[ev].mean() - Y[j, ev].mean()); viol += L[j].mean() > eps
    return dict(tok=float(np.mean(tok)), loss=float(np.mean(loss)), viol=viol / reps, certified=cert / reps)

def r2_one(args):
    (m, b), d = args; x = sub(d, m, b); out = {}
    for eps in (0.05, 0.10):
        Yd, Cd, Ld = outcome_matrix(x); YK, CK = x.YK.to_numpy(int), x.CK.to_numpy(float)
        out[str(eps)] = dict(det=evaluate(Yd, Cd, Ld, YK, CK, eps), mix=evaluate(*mixtures(x), eps), untied=untied_eval(x, eps))
        for k in out[str(eps)].values(): k.pop('picked', None)
    return f'{m}|{b}', out

def r2_toy():
    """Reviewer's two-level example and a three-level example of the cost of tying t0 = t1."""
    eps = 0.05
    two = dict(deterministic=2.0, randomised=round(eps * 1 + (1 - eps) * 2, 4))
    # three levels, two score groups (prob 1/2 each). Group A: L0 right (cost 1); group B: L0 wrong,
    # L1 right (cum. cost 2); top always right (cum. cost 4). Scores: S0 = 0.9 (A) / 0.1 (B);
    # S1 = 0.2 (A, unused) / 0.8 (B). Optimal: stop A at L0, B at L1 -> cost 1.5, risk 0.
    # Tied threshold t: t < 0.1 stops everything at L0 (risk 0.5); 0.1 <= t < 0.8 stops A at L0, B at L1
    # (cost 1.5) -- tying is harmless here. With S1(B) = 0.05 instead, the tied rule must choose t < 0.05
    # (risk 0.5) or escalate B to the top (cost 0.5*1 + 0.5*4 = 2.5); untied t0 = 0.5, t1 = 0.0 gives 1.5.
    three = dict(untied=1.5, tied=2.5, tied_risk_at_untied_cost=0.5)
    return dict(two_level=two, three_level=three)

# ---------------- R3: uncertainty of the monotone-gain diagnostic ----------------
def r3_one(args):
    (m, b), d, t1 = args; x = sub(d, m, b); rng = np.random.default_rng(0); out = {}
    for k in (0, 1):
        S = x[f'S{k}'].to_numpy(float); g = (x.YK - x[f'Y{k}']).to_numpy(float)
        q = np.searchsorted(np.quantile(S, [0.2, 0.4, 0.6, 0.8]), S, side='right')
        mu = [g[q == j].mean() for j in range(5)]; n = [int((q == j).sum()) for j in range(5)]
        B = []; R = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x)); gi, qi = g[i], q[i]
            mj = [gi[qi == j].mean() if (qi == j).any() else np.nan for j in range(5)]; B.append(mj)
            R.append(spearmanr(range(5), mj)[0] if not np.isnan(mj).any() else np.nan)
        B = np.array(B); R = np.array(R)
        r = dict(n=n, gain=[100 * v for v in mu], lo=list(100 * np.nanquantile(B, .025, 0)), hi=list(100 * np.nanquantile(B, .975, 0)),
                 rho=float(spearmanr(range(5), mu)[0]), rho_ci=[float(np.nanquantile(R, .025)), float(np.nanquantile(R, .975))])
        if k == 0:                                                       # continuation that may stop at L1
            cont = np.where((x.S1 > t1) & (x.H1 <= TAU), x.Y1, x.Y2).astype(float)
            gc = cont - x.Y0.to_numpy(float); gf = (x.Y2 - x.Y0).to_numpy(float)
            r['opt_cont'] = [100 * gc[q == j].mean() for j in range(5)]; r['fixed_terminal'] = [100 * gf[q == j].mean() for j in range(5)]
            r['t1'] = t1
        out[f'L{k}'] = r
    return f'{m}|{b}', out

# ---------------- R4: simple baselines under the same certification ----------------
def r4_one(args):
    (m, b), d = args; x = sub(d, m, b); u = np.random.default_rng(1).random(len(x)); out = {}
    P = [round(0.1 * i, 1) for i in range(1, 10)]                         # share sent to the cheap rung, conservative first
    fams = {'always_L0': rows(x, [x.Y0.to_numpy()], [x.C0.to_numpy(float)]),
            'always_L1': rows(x, [x.Y1.to_numpy()], [x.C1.to_numpy(float)]),
            'mix_L0': rows(x, [np.where(u < p, x.Y0, x.YK) for p in P], [np.where(u < p, x.C0, x.CK) for p in P]),
            'mix_L1': rows(x, [np.where(u < p, x.Y1, x.YK) for p in P], [np.where(u < p, x.C1, x.CK) for p in P]),
            'two_level_L0_ref': rows(x, *map(list, zip(*(ladder(x, t, t, top=('YK', 'CK'), rungs=(0,)) for t in GRID)))),
            'climb': None}
    for eps in (0.05, 0.10):
        o = {}
        for k, f in fams.items():
            if f is None: Y, C, L = outcome_matrix(x); f = (Y, C, L, x.YK.to_numpy(int), x.CK.to_numpy(float))
            r = evaluate(*f, eps); r.pop('picked'); o[k] = r
        out[str(eps)] = o
    return f'{m}|{b}', out

# ---------------- R5: calibration cost and payback ----------------
def r5(d):
    out = {}
    for m, b in SETTINGS:
        x = sub(d, m, b); Y, C, L, YK, CK = (*outcome_matrix(x), x.YK.to_numpy(int), x.CK.to_numpy(float))
        n_cal = len(x) // 2; Tcal = n_cal * float((x.C2 + x.CK).mean()); o = dict(n_cal=n_cal, T_cal=Tcal)
        for eps in (0.05, 0.10):
            sav, calls, ref_calls = [], [], []
            for cal, ev in splits(len(x)):
                j = calibrate(L[:, cal], C[:, cal], eps, DELTA); sav.append(CK[ev].mean() - C[j, ev].mean())
                if j == 0: calls.append(x.NK[ev].mean())
                else:
                    t = GRID[j - 1]; s0 = (x.S0 > t) & (x.H0 <= TAU); s1 = ~s0 & (x.S1 > t) & (x.H1 <= TAU)
                    calls.append(np.where(s0, x.N0, np.where(s1, x.N1, x.N2))[ev].mean())
            D = float(np.mean(sav))
            o[str(eps)] = dict(saving=D, payback=(Tcal / D if D > 0 else None), calls=float(np.mean(calls)))
        o.update(calls_ref=float(x.NK.mean()), calls_full_climb=float(x.N2.mean()),
                 sec_ref=[float(x.TK.mean()), float(x.TK.median()), float(x.TK.quantile(.95))],
                 sec_full_climb=[float(x.T2.mean()), float(x.T2.median()), float(x.T2.quantile(.95))])
        out[f'{m}|{b}'] = o
    return out

# ---------------- R7: consistency checks ----------------
def r7(d):
    out = {}
    for m, b in SETTINGS:
        x = sub(d, m, b)
        veto0 = (x.H0 > TAU); esc = veto0 & (x.H1 > TAU)                   # at t = -1e9 only the entropy gate escalates
        ties = int(sum(np.isclose(x[f'S{k}'].to_numpy(float)[:, None], np.array(GRID)[None, :], atol=0, rtol=0).sum() for k in (0, 1)))
        out[f'{m}|{b}'] = dict(endpoint_stop_L0=100 * float((~veto0).mean()), endpoint_stop_L1=100 * float((veto0 & ~(x.H1 > TAU)).mean()),
                               endpoint_escalate=100 * float(esc.mean()), ties=ties)
    return out

# ---------------- R4 diagnostics: grid resolution and escalation overhead ----------------
COARSE = [-0.01, -0.02, -0.03, -0.05, -0.075, -0.1, -0.15, -0.2, -0.3, -0.5, -1.0, -1e9]
def r4diag_one(args):
    """CLIMB with a coarse 12-point grid, and a counterfactual in which an escalated task pays only
    the reference pipeline (no sunk cost of failed cheap attempts, no seeding overhead)."""
    (m, b), d = args; x = sub(d, m, b); Y, C, L = outcome_matrix(x); YK, CK = x.YK.to_numpy(int), x.CK.to_numpy(float)
    coarse = rows(x, *map(list, zip(*(ladder(x, t, t) for t in COARSE))))
    free = rows(x, *map(list, zip(*(ladder(x, t, t, top=('YK', 'CK')) for t in GRID))))
    free = (free[0], np.array([free[1][0]] + [c - x.C0.to_numpy(float) * ((c != x.C0.to_numpy(float)) & (c != x.C1.to_numpy(float)))
                                              for c in free[1][1:]]), *free[2:])
    out = {}
    for eps in (0.05, 0.10):
        o = {k: evaluate(*f, eps) for k, f in (('fine', (Y, C, L, YK, CK)), ('coarse', coarse), ('free_escalation', free))}
        for r in o.values(): r.pop('picked')
        out[str(eps)] = o
    return f'{m}|{b}', out

def full_pool_t1(d, m, b, eps=0.05):
    x = sub(d, m, b); Y, C, L = outcome_matrix(x); j = calibrate(L, C, eps, DELTA)
    return GRID[j - 1] if j else 1e9

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'; d = load()
    if what in ('r1', 'all'): dump('r1', r1(d))
    if what in ('r5', 'all'): dump('r5', r5(d))
    if what in ('r7', 'all'): dump('r7', r7(d))
    with Pool(8) as p:
        if what in ('r2', 'all'): dump('r2', dict(toy=r2_toy(), replay=dict(p.map(r2_one, [(s, d) for s in SETTINGS]))))
        if what in ('r3', 'all'): dump('r3', dict(p.map(r3_one, [(s, d, full_pool_t1(d, *s)) for s in SETTINGS])))
        if what in ('r4', 'all'): dump('r4', dict(p.map(r4_one, [(s, d) for s in SETTINGS])))
        if what in ('r4diag', 'all'): dump('r4diag', dict(p.map(r4diag_one, [(s, d) for s in SETTINGS])))
    print('done', what)
