"""Escalation on a learned estimate of the signed gain instead of the cheap rung's log-probability.
Proposition 1 makes the optimal policy a threshold on the gain g(s) = E[Y_ref - Y_stop | s], so the
score here is a classifier's estimate of E[Y_ref - Y_0 | features] = P(rescue) - P(harm), learned on a
training split and frozen before calibration. Decision: stop at L0 when the estimated gain is low,
otherwise run the reference from scratch (cost C0 + C_ref).
Features, all from the cheap rungs' own records (no extra calls): S0, S1, L0 generation length,
L1 extra tokens, question length (L0 prompt tokens), and whether L0 and L1 give the same answer.
Per split: train 40% / calibrate 30% / test 30%. The candidate thresholds are quantiles of the
training scores (fixed before calibration), tested in order by CLIMB's fixed-sequence betting test.
Baseline: the log-probability threshold S0 on the same test split, calibrated on train + calibrate.
usage: python analysis/learned_gain.py MODEL:BENCH:climb_trace ... (records from data/records_new.csv)"""
import json, os, re, sys, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import calibrate, GRID, HERE

EPS = (0.02, 0.05, 0.10); DELTA = 0.1; SPLITS = int(os.environ.get('SPLITS', 100))
ROOT = os.path.join(HERE, '..', '..')


def _ans(s, bcb):
    s = s or ''
    if bcb:
        m = re.findall(r'```python\s*\n(.*?)```', s, re.S)
        return re.sub(r'\s+', '', m[-1]) if m else re.sub(r'\s+', '', s)
    m = re.findall(r'ANSWER:\s*(.+)', s)
    return m[-1].strip().strip('*$. ').lower() if m else None


def features(x, trace):
    T = {}
    for l in open(os.path.join(ROOT, trace)):
        r = json.loads(l); ex = [e for e in r['rounds'] if e['role'] == 'executor']
        if len(ex) >= 2:
            T[r['task_id']] = ex
    bcb = 'bcb' in trace
    rows = []
    for tid in x.task_id:
        e0, e1 = T[tid][0], T[tid][1]
        a0, a1 = _ans(e0.get('answer_snapshot'), bcb), _ans(e1.get('answer_snapshot'), bcb)
        rows.append([e0.get('prompt_tokens') or 0, e0.get('completion_tokens') or 0, float(a0 is not None and a0 == a1)])
    F = np.column_stack([x.S0.fillna(-9), x.S1.fillna(-9), np.array(rows), x.C1 - x.C0])
    return F


def candidates(score, grid, Y0, YK, C0, CK):
    """Row 0 = reference; row m stops at L0 when score > grid[m-1] (grid descending)."""
    stop = score[None, :] > np.asarray(grid)[:, None]
    Y = np.vstack([YK, np.where(stop, Y0, YK)]); C = np.vstack([CK, np.where(stop, C0, C0 + CK)])
    return Y, C, YK[None, :] - Y


def run(x, F):
    Y0, YK = x.Y0.to_numpy(int), x.YK.to_numpy(int); C0, CK = x.C0.to_numpy(float), x.CK.to_numpy(float)
    gain = YK - Y0; S0 = x.S0.fillna(-np.inf).to_numpy(float); N = len(x); rng = np.random.default_rng(0)
    out = {str(e): dict(learned=[], learned_v=0, logprob=[], logprob_v=0) for e in EPS}
    for _ in range(SPLITS):
        p = rng.permutation(N); a, b = int(.4 * N), int(.7 * N); tr, ca, te = p[:a], p[a:b], p[b:]
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
        clf.fit(F[tr], gain[tr]); cls = list(clf.classes_)
        P = clf.predict_proba(F)
        g = (P[:, cls.index(1)] if 1 in cls else 0) - (P[:, cls.index(-1)] if -1 in cls else 0)
        sc = -g                                                   # high score = small gain = stop
        grid = list(np.unique(np.quantile(sc[tr], np.linspace(0.99, 0.0, 40)))[::-1]) + [-np.inf]
        Y, C, L = candidates(sc, grid, Y0, YK, C0, CK)
        Yb, Cb, Lb = candidates(S0, list(GRID), Y0, YK, C0, CK)
        R, Rb = L.mean(1), Lb.mean(1)
        for e in EPS:
            j = calibrate(L[:, ca], C[:, ca], e, DELTA); o = out[str(e)]
            o['learned'].append(C[j, te].mean() / CK[te].mean()); o['learned_v'] += R[j] > e
            cb = ca if os.environ.get("SAME_CAL") else np.concatenate([tr, ca]); jb = calibrate(Lb[:, cb], Cb[:, cb], e, DELTA)
            o['logprob'].append(Cb[jb, te].mean() / CK[te].mean()); o['logprob_v'] += Rb[jb] > e
    return {e: dict(learned=100 * np.mean(o['learned']), learned_viol=100 * o['learned_v'] / SPLITS,
                    logprob=100 * np.mean(o['logprob']), logprob_viol=100 * o['logprob_v'] / SPLITS) for e, o in out.items()}


if __name__ == '__main__':
    d = pd.read_csv(os.path.join(HERE, '..', 'data', 'records_new.csv'))
    for spec in sys.argv[1:]:
        m, b, tr = spec.split(':')
        x = d[(d.model == m) & (d.bench == b)].reset_index(drop=True)
        r = run(x, features(x, tr))
        print(f"{m} {b} n={len(x)} | tokens % (violation %), learned gain vs log-prob S0: " +
              ' | '.join(f"e{float(e)*100:.0f}: {v['learned']:.0f} ({v['learned_viol']:.0f}) vs {v['logprob']:.0f} ({v['logprob_viol']:.0f})" for e, v in r.items()))
