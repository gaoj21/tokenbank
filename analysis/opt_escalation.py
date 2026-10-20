"""Escalation as constrained optimisation, evaluated offline against the log-prob threshold gate.

Problem: choose a stopping rung pi(x) in {L0, L1, pipeline} to minimise expected cost subject
to accuracy loss vs the pipeline <= eps. Lagrangian relaxation gives a sequential rule: at each
rung stop if the predicted probability of being right now is at least the predicted probability
after escalating minus lam times the extra expected cost.
  stage A (after L0; features of L0):       predict p0, p1, pP  -> stop / continue
  stage B (after L1; features of L0 and L1): predict p1, pP     -> stop / escalate
Predictors are logistic regressions, trained per model on all four benchmarks pooled (benchmark
one-hot as a feature). lam is chosen by Learn-then-Test on a separate calibration split, with
the same exact-binomial harm test and 'direct pipeline' fallback as analysis/risk_gate.py, so
both methods carry the same guarantee and are compared on tokens at equal eps.
Splits: 40% train, 20% calibrate, 40% test, 200 repetitions, stratified by benchmark.
"""
import json, random, statistics as st, warnings
import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items, apply as gate_apply, GRID as GATE_GRID, top
BENCH = ['MATH', 'GAIA', 'FRAMES', 'MuSiQue']
def norm(a): return ''.join(ch for ch in (a or '').lower() if ch.isalnum())[-40:]
def extract(a):
    """Final answer as the harness extracts it (ANSWER: line or last non-empty line), normalised."""
    a = a or ''
    m = [l for l in a.splitlines() if l.strip().upper().startswith('ANSWER:')]
    t = m[-1].split(':', 1)[1] if m else (next((l for l in reversed(a.splitlines()) if l.strip()), ''))
    return ''.join(ch for ch in t.lower() if ch.isalnum())
def load_family(fam):
    rows = []
    for ds, ae, nf in CELLS[fam]:
        agree = {}
        import risk_gate as _rg
        src = _rg._climbx_path(ae) if _rg.SOURCE == 'climbx' else ae
        for l in open(src):
            d = json.loads(l)
            if d['config'].endswith('AllEsc') or d['config'] == 'CLIMB-Live':
                rs = [r for r in d['rounds'] if r['role'] == 'executor']
                if len(rs) >= 2:
                    agree[d['task_id']] = extract(rs[0].get('answer_snapshot')) == extract(rs[1].get('answer_snapshot'))
        for x in items(ae, nf, keep_id=True):
            x['ds'] = ds; x['agree'] = agree.get(x['id'], False); rows.append(x)
    return rows
def featA(x):
    oh = [1.0 if x['ds'] == b else 0.0 for b in BENCH]
    return [x['lp0'] if x['lp0'] is not None else -1, x['e0'] if x['e0'] is not None else 1, x['c0'] / 1000] + oh
def featB(x):
    return featA(x) + [x['lp1'] if x['lp1'] is not None else -1, x['e1'] if x['e1'] is not None else 1,
                       (x['c1'] - x['c0']) / 1000, 1.0 if x['agree'] else 0.0]
def fit(X, y):
    m = LogisticRegression(max_iter=2000, C=1.0)
    if len(set(y)) < 2: return lambda Z: np.full(len(Z), float(y[0]))
    m.fit(X, y); return lambda Z: m.predict_proba(Z)[:, 1]
def make_policy(train):
    XA = np.array([featA(x) for x in train]); XB = np.array([featB(x) for x in train])
    pA0, pA1, pAP = fit(XA, [x['ok0'] for x in train]), fit(XA, [x['ok1'] for x in train]), fit(XA, [x['okP'] for x in train])
    pB1, pBP = fit(XB, [x['ok1'] for x in train]), fit(XB, [x['okP'] for x in train])
    cost = {b: (st.mean([x['c1'] - x['c0'] for x in train if x['ds'] == b] or [0]), st.mean([top(x)[1] - x['c1'] for x in train if x['ds'] == b] or [0])) for b in BENCH}
    def decide(S):
        A = np.array([featA(x) for x in S]); B = np.array([featB(x) for x in S])
        return pA0(A), pA1(A), pAP(A), pB1(B), pBP(B), [cost[x['ds']] for x in S]
    return decide
def run_policy(S, pred, lam):
    if lam == 'direct': return [(x['okP'], x['cP']) for x in S]
    a0, a1, aP, b1, bP, cs = pred
    out = []
    for i, x in enumerate(S):
        d1, dP = cs[i]
        if a0[i] >= max(a1[i] - lam * d1, aP[i] - lam * (d1 + dP)): out.append((x['ok0'], x['c0'])); continue
        if b1[i] >= bP[i] - lam * dP: out.append((x['ok1'], x['c1'])); continue
        out.append(top(x))
    return out
LAMS = ['direct'] + [0.0] + list(np.geomspace(1e-6, 1e-2, 60))
def ltt(cal, evalf, grid, eps, delta=0.1):
    n = len(cal); certified = ['direct']
    for lam in grid[1:]:
        res = evalf(cal, lam)
        k = sum(1 for x, (ok, c) in zip(cal, res) if x['okP'] and not ok)
        if binom.cdf(k, n, eps) <= delta: certified.append(lam)
        else: break
    return min(certified, key=lambda lam: st.mean(c for _, c in evalf(cal, lam)))
if __name__ == '__main__':
    random.seed(1)
    summary = {}
    for fam in CELLS:
        rows = load_family(fam)
        for eps in (0.02, 0.05):
            acc = {b: {'opt': [], 'gate': [], 'ropt': [], 'rgate': [], 'vopt': 0, 'vgate': 0} for b in BENCH}
            for rep in range(200):
                tr, ca, te = [], [], []
                for b in BENCH:
                    g = [x for x in rows if x['ds'] == b]; random.shuffle(g)
                    n = len(g); tr += g[:int(.4*n)]; ca += g[int(.4*n):int(.6*n)]; te += g[int(.6*n):]
                pol = make_policy(tr)
                evalO = lambda S, lam: run_policy(S, pol(S), lam)
                evalG = lambda S, lam: [gate_apply(x, lam) for x in S]
                for b in BENCH:
                    cab = [x for x in ca if x['ds'] == b]; teb = [x for x in te if x['ds'] == b]
                    lo = ltt(cab, evalO, LAMS, eps); lg = ltt(cab, evalG, GATE_GRID, eps)
                    for key, lam, ev in (('opt', lo, evalO), ('gate', lg, evalG)):
                        res = ev(teb, lam)
                        r = st.mean(int(x['okP']) - int(ok) for x, (ok, c) in zip(teb, res))
                        acc[b][key].append(st.mean(c for _, c in res) / st.mean(x['cP'] for x in teb))
                        acc[b]['r' + key].append(r); acc[b]['v' + key] += r > eps
            for b in BENCH:
                a = acc[b]
                summary[(fam, b, eps)] = a
                print(f"{fam:<9}{b:<8} eps={100*eps:.0f}%   opt: tokens {100*st.mean(a['opt']):5.0f}%  risk {100*st.mean(a['ropt']):+5.2f}  viol {a['vopt']/2:4.1f}%"
                      f"   | gate: tokens {100*st.mean(a['gate']):5.0f}%  risk {100*st.mean(a['rgate']):+5.2f}  viol {a['vgate']/2:4.1f}%")
    json.dump({f'{k[0]}|{k[1]}|{k[2]}': {kk: (st.mean(v) if isinstance(v, list) else v) for kk, v in a.items()} for k, a in summary.items()},
              open('results/opt_escalation.json', 'w'), indent=1)
