"""Four escalation policies under the same risk control, offline.
A shared log-prob threshold (current); B separate L0/L1 thresholds (offset chosen on train);
C L0 threshold + stop at L1 when L1's extracted answer equals L0's; D non-parametric Bayes rule:
per benchmark and rung, P(correct) is estimated by log-prob decile on the train split, and the
Lagrangian rule (stop if p_now >= p_next - lam * extra cost) is applied. Each family is reduced
to a 1-D sequence ordered from conservative to aggressive and passed to the same Learn-then-Test
selector (exact binomial on harm, 'direct pipeline' fallback, cheapest certified setting).
Splits per benchmark: 40% train, 20% calibrate, 40% test; 200 repetitions; same splits for all."""
import json, random, statistics as st, sys, bisect
import numpy as np
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, TAU, top
from opt_escalation import load_family, ltt, BENCH
conf = lambda lp, e, th: lp is not None and lp > th and (e is None or e <= TAU)
def pol_A(x, th):
    if th == 'direct': return x['okP'], x['cP']
    if conf(x['lp0'], x['e0'], th): return x['ok0'], x['c0']
    if conf(x['lp1'], x['e1'], th): return x['ok1'], x['c1']
    return top(x)
def mk_B(off):
    def f(x, th):
        if th == 'direct': return x['okP'], x['cP']
        if conf(x['lp0'], x['e0'], th): return x['ok0'], x['c0']
        if conf(x['lp1'], x['e1'], th + off): return x['ok1'], x['c1']
        return top(x)
    return f
def pol_C(x, th):
    if th == 'direct': return x['okP'], x['cP']
    if conf(x['lp0'], x['e0'], th): return x['ok0'], x['c0']
    if x['agree']: return x['ok1'], x['c1']
    return top(x)
TH = ['direct'] + [round(-0.005 * i, 3) for i in range(1, 121)]
def mk_D(train):
    tab = {}
    for b in BENCH:
        g = [x for x in train if x['ds'] == b]
        if not g: continue
        e0 = sorted(x['lp0'] if x['lp0'] is not None else -9 for x in g); e1 = sorted(x['lp1'] if x['lp1'] is not None else -9 for x in g)
        q0 = [e0[int(len(e0) * k / 10)] for k in range(1, 10)]; q1 = [e1[int(len(e1) * k / 10)] for k in range(1, 10)]
        def rate(keyf, qs, lab):
            bins = [[] for _ in range(10)]
            for x in g: bins[bisect.bisect(qs, keyf(x))].append(int(x[lab]))
            pri = st.mean(int(x[lab]) for x in g)
            return [(sum(bn) + 2 * pri) / (len(bn) + 2) for bn in bins]      # shrink to the mean
        k0 = lambda x: x['lp0'] if x['lp0'] is not None else -9
        k1 = lambda x: x['lp1'] if x['lp1'] is not None else -9
        tab[b] = dict(q0=q0, q1=q1, p0=rate(k0, q0, 'ok0'), p1a=rate(k0, q0, 'ok1'), pPa=rate(k0, q0, 'okP'),
                      p1=rate(k1, q1, 'ok1'), pPb=rate(k1, q1, 'okP'),
                      d1=st.mean(x['c1'] - x['c0'] for x in g), dP=st.mean(top(x)[1] - x['c1'] for x in g))
    def f(x, lam):
        if lam == 'direct': return x['okP'], x['cP']
        t = tab[x['ds']]
        i0 = bisect.bisect(t['q0'], x['lp0'] if x['lp0'] is not None else -9)
        if t['p0'][i0] >= max(t['p1a'][i0] - lam * t['d1'], t['pPa'][i0] - lam * (t['d1'] + t['dP'])): return x['ok0'], x['c0']
        i1 = bisect.bisect(t['q1'], x['lp1'] if x['lp1'] is not None else -9)
        if t['p1'][i1] >= t['pPb'][i1] - lam * t['dP']: return x['ok1'], x['c1']
        return top(x)
    return f
LAMS = ['direct', 0.0] + list(np.geomspace(1e-6, 1e-2, 60))
def best_offset(train, eps):
    best, bc = 0.0, 1e18
    for off in (-0.10, -0.05, 0.0, 0.05, 0.10):
        f = mk_B(off)
        for th in TH[1:]:
            r = st.mean(int(x['okP']) - int(f(x, th)[0]) for x in train)
            if r <= eps:
                c = st.mean(f(x, th)[1] for x in train)
                if c < bc: best, bc = off, c
    return best
if __name__ == '__main__':
    random.seed(2)
    res = {}
    for fam in CELLS:
        rows = load_family(fam)
        for eps in (0.05,):
            acc = {(b, m): [] for b in BENCH for m in 'ABCD'}; viol = {(b, m): 0 for b in BENCH for m in 'ABCD'}
            for rep in range(200):
                tr, ca, te = [], [], []
                for b in BENCH:
                    g = [x for x in rows if x['ds'] == b]; random.shuffle(g); n = len(g)
                    tr += g[:int(.4*n)]; ca += g[int(.4*n):int(.6*n)]; te += g[int(.6*n):]
                fD = mk_D(tr)
                for b in BENCH:
                    trb = [x for x in tr if x['ds'] == b]; cab = [x for x in ca if x['ds'] == b]; teb = [x for x in te if x['ds'] == b]
                    fB = mk_B(best_offset(trb, eps))
                    for m, f, grid in (('A', pol_A, TH), ('B', fB, TH), ('C', pol_C, TH), ('D', fD, LAMS)):
                        ev = lambda S, th, f=f: [f(x, th) for x in S]
                        th = ltt(cab, ev, grid, eps)
                        out = ev(teb, th)
                        r = st.mean(int(x['okP']) - int(ok) for x, (ok, c) in zip(teb, out))
                        acc[(b, m)].append(st.mean(c for _, c in out) / st.mean(x['cP'] for x in teb)); viol[(b, m)] += r > eps
            for b in BENCH:
                line = f"{fam:<9}{b:<8} eps=5%  " + "  ".join(f"{m}: {100*st.mean(acc[(b, m)]):4.0f}% (viol {viol[(b, m)]/2:.1f}%)" for m in 'ABCD')
                print(line); res[f'{fam}|{b}'] = {m: st.mean(acc[(b, m)]) for m in 'ABCD'}
    json.dump(res, open('results/escalation_variants.json', 'w'), indent=1)
