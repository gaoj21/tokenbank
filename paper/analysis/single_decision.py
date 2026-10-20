"""CLIMB as deployed (a ladder that escalates to the reference) and its one-decision variant on
data/records.csv, with the checks of the theory.

Design: the cheap stage (L0 then L1) always runs and returns L1's answer (Y1, cost C1); one decision
on the score S either stops or runs the reference pipeline (YK, total cost C1 + CK); the empty
action runs the reference directly (YK, CK). Default score: S = S1, the mean log-prob of L1's
final call. Checks, per setting:
  (M)        gain g = E[YK - Y1 | S] and gain per token g/c by score quintile, with Spearman rho;
  prefix     whether {t : r(t) <= eps} is an initial segment of the threshold sequence (full pool);
  efficiency CLIMB's cost against the best grid threshold on the full pool (J(t_eps)) and against
             the optimal policy of any form at that tolerance (fractional-knapsack bound on score
             bins), as the calibration set grows.
Also compares the cost of single-decision CLIMB with the two-decision ladder and with L0 -> reference.
Usage: python analysis/single_decision.py [records.csv] -> data/single_decision.json"""
import json, os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import outcome_matrix, calibrate, GRID, HERE

EPS = (0.02, 0.05, 0.10); DELTA = 0.1; SPLITS = 200; NS = (50, 100, 200, 400, 800)

def single(x, score):
    """Candidate matrices for the single decision: row 0 = reference, then GRID thresholds."""
    YK, CK = x.YK.to_numpy(int), x.CK.to_numpy(float); Y1, C1 = x.Y1.to_numpy(int), x.C1.to_numpy(float)
    S = np.nan_to_num(score.to_numpy(float), nan=-np.inf)                 # no log-probs: escalate
    stop = S[None, :] > np.array(GRID)[:, None]
    Y = np.vstack([YK, np.where(stop, Y1, YK)]); C = np.vstack([CK, np.where(stop, C1, C1 + CK)])
    return Y, C, YK[None, :] - Y, YK, CK

def ladder_ref(x):
    """Deployed ladder: stop at L0 if S0 > t, else at L1 if S1 > t, else run the reference from
    scratch (YK, cost C1 + CK). Row 0 = reference directly."""
    YK, CK = x.YK.to_numpy(int), x.CK.to_numpy(float); G = np.array(GRID)[:, None]
    S0 = np.nan_to_num(x.S0.to_numpy(float), nan=-np.inf); S1 = np.nan_to_num(x.S1.to_numpy(float), nan=-np.inf)
    s0 = S0[None, :] > G; s1 = ~s0 & (S1[None, :] > G)
    Y = np.vstack([YK, np.where(s0, x.Y0, np.where(s1, x.Y1, YK))])
    C = np.vstack([CK, np.where(s0, x.C0, np.where(s1, x.C1, x.C1.to_numpy(float) + CK))])
    return Y, C, YK[None, :] - Y, YK, CK

def prefix_check(mats):
    R = mats[2].mean(1)[1:]; out = {}
    for e in EPS:
        okm = R <= e; first_bad = int(np.argmax(~okm)) if (~okm).any() else len(okm)
        out[str(e)] = dict(is_prefix=bool(not okm[first_bad:].any()), safe=int(okm.sum()), prefix_len=first_bad,
                           lost=int(okm[first_bad:].sum()))
    return out

def two_level(x):
    YK, CK = x.YK.to_numpy(int), x.CK.to_numpy(float)
    S = np.nan_to_num(x.S0.to_numpy(float), nan=-np.inf); stop = S[None, :] > np.array(GRID)[:, None]
    Y = np.vstack([YK, np.where(stop, x.Y0, YK)]); C = np.vstack([CK, np.where(stop, x.C0, x.C0.to_numpy(float) + CK)])
    return Y, C, YK[None, :] - Y, YK, CK

def replay(mats, n=None, reps=SPLITS, seed=0):
    Y, C, L, YK, CK = mats; N = L.shape[1]; R = L.mean(1); rng = np.random.default_rng(seed); out = {}
    for e in EPS:
        tok, viol = [], 0
        for _ in range(reps):
            p = rng.permutation(N); k = n or N // 2; cal, ev = p[:k], p[k:]
            j = calibrate(L[:, cal], C[:, cal], e, DELTA)
            tok.append(C[j, ev].mean() / CK[ev].mean()); viol += R[j] > e
        out[str(e)] = dict(tok=100 * float(np.mean(tok)), viol=100 * viol / reps)
    return out

def climb_families(x):
    """Deployed CLIMB: the depth of the ladder is chosen on the calibration data. Two ordered
    families, the full ladder (L0 -> L1 -> reference) and L0 -> reference, are each certified at
    delta / 2 and the one with the lower calibration cost is deployed; by a union bound both
    certificates hold together with probability at least 1 - delta, so the choice keeps the guarantee."""
    return [ladder_ref(x), two_level(x)]

def calibrate_multi(fams, cal, e, delta=DELTA):
    best = None
    for f, (Y, C, L, YK, CK) in enumerate(fams):
        j = calibrate(L[:, cal], C[:, cal], e, delta / len(fams)); c = C[j, cal].mean()
        if best is None or c < best[0]: best = (c, f, j)
    return best[1], best[2]

def replay_multi(fams, n=None, reps=SPLITS, seed=0):
    N = fams[0][2].shape[1]; CK = fams[0][4]; rng = np.random.default_rng(seed); out = {}
    for e in EPS:
        tok, viol, depth = [], 0, 0
        for _ in range(reps):
            p = rng.permutation(N); k = n or N // 2; cal, ev = p[:k], p[k:]
            f, j = calibrate_multi(fams, cal, e); C, L = fams[f][1], fams[f][2]
            tok.append(C[j, ev].mean() / CK[ev].mean()); viol += L[j].mean() > e; depth += f == 0
        out[str(e)] = dict(tok=100 * float(np.mean(tok)), viol=100 * viol / reps, ladder_share=100 * depth / reps)
    return out

def oracle_threshold(mats, e):
    """Cheapest candidate (reference or grid threshold) whose full-pool risk is at most eps."""
    Y, C, L, YK, CK = mats; R = L.mean(1); Cm = C.mean(1); ok = R <= e
    return 100 * float(Cm[ok].min() / CK.mean())

def knapsack_bound(x, score, e, bins=20):
    """Optimal policy of any form on score bins (fractional knapsack, randomising one bin),
    compared with running the reference directly; full-pool quantities."""
    S = np.nan_to_num(score.to_numpy(float), nan=-np.inf); q = pd.qcut(pd.Series(S).rank(method='first'), bins, labels=False)
    g = (x.YK - x.Y1).groupby(q).sum().to_numpy() / len(x); c = x.CK.groupby(q).sum().to_numpy() / len(x)
    order = np.argsort(g / c); risk, saved = 0.0, 0.0
    for i in order:
        if g[i] <= 0 or risk + g[i] <= e: risk += g[i]; saved += c[i]
        else: saved += c[i] * (e - risk) / g[i]; break
    J = x.C1.mean() + x.CK.mean() - saved
    return 100 * float(min(J, x.CK.mean()) / x.CK.mean())

def checks(x, score):
    S = score.to_numpy(float); ok = ~np.isnan(S); xs = x[ok]; s = S[ok]
    q = pd.qcut(pd.Series(s).rank(method='first'), 5, labels=False).to_numpy()
    g = [float((xs.YK - xs.Y1).to_numpy()[q == k].mean()) for k in range(5)]
    c = [float(xs.CK.to_numpy()[q == k].mean()) for k in range(5)]
    gc = [a / b for a, b in zip(g, c)]
    mats = single(x, score); R = mats[2].mean(1)[1:]                        # risk along the grid
    prefix = {}
    for e in EPS:
        okm = R <= e; first_bad = np.argmax(~okm) if (~okm).any() else len(okm)
        prefix[str(e)] = dict(is_prefix=bool(not okm[first_bad:].any()), safe=int(okm.sum()),
                              prefix_len=int(first_bad), lost=int(okm[first_bad:].sum()))
    return dict(g=[100 * v for v in g], c=c, g_per_1k=[1e5 * v for v in gc],
                rho_g=float(spearmanr(range(5), g)[0]), rho_gc=float(spearmanr(range(5), gc)[0]), prefix=prefix)

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'records.csv')
    d = pd.read_csv(path); out = {}
    for (m, b), x in d.groupby(['model', 'bench'], sort=False):
        x = x.reset_index(drop=True); o = dict(n=len(x))
        lad = ladder_ref(x); mats = single(x, x.S1)
        o['checks'] = checks(x, x.S1)                                     # (M) on the one-decision score
        o['ladder_prefix'] = prefix_check(lad)
        fams = climb_families(x)
        o['climb'] = replay_multi(fams)                                   # deployed method (depth chosen on calibration)
        o['climb_ladder'] = replay(lad)                                   # full ladder only
        o['ladder_oracle_threshold'] = {str(e): min(oracle_threshold(f, e) for f in fams) for e in EPS}
        o['ladder_by_n'] = {str(n): replay_multi(fams, n=n, reps=100) for n in NS if n < len(x) * 0.8}
        o['climb_single'] = replay(mats)
        o['climb_ladder_seeded'] = replay(outcome_matrix(x) + (x.YK.to_numpy(int), x.CK.to_numpy(float)))
        o['two_level_L0'] = replay(two_level(x))
        o['score_mean'] = replay(single(x, (x.S0 + x.S1) / 2))
        o['score_min'] = replay(single(x, np.minimum(x.S0, x.S1)))
        o['oracle_threshold'] = {str(e): oracle_threshold(mats, e) for e in EPS}
        o['knapsack_bound'] = {str(e): knapsack_bound(x, x.S1, e) for e in EPS}
        o['by_n'] = {str(n): replay(mats, n=n, reps=100) for n in NS if n < len(x) * 0.8}
        out[f'{m}|{b}'] = o
        f = lambda r: ' '.join(f"{r[str(e)]['tok']:.0f}" for e in EPS)
        g = lambda r: ' '.join(f"{r[str(e)]:.0f}" for e in EPS)
        pre = all(v['is_prefix'] for v in o['ladder_prefix'].values())
        print(f"{m:9s} {b:8s} CLIMB {f(o['climb'])} [ladder chosen {' '.join(f"{o['climb'][str(e)]['ladder_share']:.0f}%" for e in EPS)}, viol {' '.join(f"{o['climb'][str(e)]['viol']:.0f}" for e in EPS)}] | ladder-only {f(o['climb_ladder'])} (best {g(o['ladder_oracle_threshold'])}, prefix {pre}) | one-decision {f(o['climb_single'])} "
              f"(best {g(o['oracle_threshold'])}, bound {g(o['knapsack_bound'])}) | seeded ladder {f(o['climb_ladder_seeded'])} | "
              f"L0->ref {f(o['two_level_L0'])} | rho_g {o['checks']['rho_g']:+.2f}")
    json.dump(out, open(os.path.join(HERE, '..', 'data', 'single_decision.json'), 'w'), indent=1)
