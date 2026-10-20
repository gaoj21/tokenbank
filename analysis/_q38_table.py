#!/usr/bin/env python
"""Consolidated Qwen3.8 results, computed from traces/ every time.

Two things this gets right that ad-hoc analysis kept getting wrong:
  1. Each trace file is read separately. Several files contain arms with the
     SAME config name (e.g. two versions of V5-Compact-Real), so merging them
     into one dict by config silently overwrites one run with the other.
  2. `stopped_by == 'budget'` is NOT the same as a budget truncation. An API
     timeout also lands there, with ~0 tokens spent and 0 rounds. Real
     truncation is separated by total_used > 0.5 * budget.
"""
import json, math, glob, collections, sys

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); ln2 = n * math.log(2)
    ls = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-ln2 for i in range(k+1)]
    m = max(ls)
    return min(1.0, 2*math.exp(m + math.log(sum(math.exp(l-m) for l in ls))))

NAMES = {'ma':'MATH','ga':'GAIA','fr':'FRAMES','mu':'MuSiQue'}
OURS, FMT, UNC = 'V5.1-NoPool', 'V5-Unc-Fmt', 'V5-Unconstrained'

rows = []
for f in sorted(glob.glob('traces/tasks_sweep_Qwen3.8-27B_q38_*.jsonl')):
    tag = f.split('_q38_')[-1].replace('.jsonl','')
    ds, budget = NAMES.get(tag[:2], tag[:2]), tag[2:] + 'k'
    by, tok, trunc, timeout = {}, {}, collections.Counter(), collections.Counter()
    for ln in open(f):
        r = json.loads(ln); c = r['config']
        by.setdefault(c, {})[r['task_id']] = bool(r['success'])
        tok.setdefault(c, {})[r['task_id']] = r['total_used']
        if r.get('stopped_by') == 'budget':
            (trunc if r['total_used'] > 0.5*r['budget'] else timeout)[c] += 1
    if OURS not in by or FMT not in by: continue
    ids = sorted(set(by[OURS]) & set(by[FMT]))
    cw = sum(1 for i in ids if by[FMT][i] and not by[OURS][i])
    wc = sum(1 for i in ids if not by[FMT][i] and by[OURS][i])
    sa = sum(by[OURS][i] for i in ids)/len(ids)*100
    sb = sum(by[FMT][i] for i in ids)/len(ids)*100
    su = sum(by[UNC][i] for i in ids)/len(ids)*100 if UNC in by else float('nan')
    dt = (sum(tok[OURS][i] for i in ids)/sum(tok[FMT][i] for i in ids)-1)*100
    rows.append((ds, budget, len(ids), sa, sb, su, sa-sb, wc, cw,
                 mcnemar(cw, wc), dt, trunc[FMT], trunc[OURS], timeout[OURS]))

order = {'MATH':0,'GAIA':1,'FRAMES':2,'MuSiQue':3}
rows.sort(key=lambda r: (r[1], order.get(r[0], 9)))
print(f'{"set":8}{"B":>5}{"n":>6}{"Ours":>7}{"+Fmt":>7}{"Naive":>7}{"Δ":>7}{"W/L":>9}{"p":>10}{"tok":>7}{"trunc":>9}{"t/o":>5}')
print('-'*90)
for ds,b,n,sa,sb,su,d,wc,cw,p,dt,tf,to_,tmo in rows:
    print(f'{ds:8}{b:>5}{n:>6}{sa:6.1f}%{sb:6.1f}%{su:6.1f}%{d:+7.1f}{f"{wc}/{cw}":>9}{p:10.1e}{dt:+6.0f}%{f"{tf}->{to_}":>9}{tmo:5}')
print('\ntrunc = real budget truncations (>50% of budget spent), baseline -> ours')
print('t/o   = ours labelled budget but actually an API timeout (<50% spent); counted as failures')


def latex():
    """Emit Table 1 in the paper's existing layout, from the same numbers above."""
    cell = {}
    for f in sorted(glob.glob('traces/tasks_sweep_Qwen3.8-27B_q38_*.jsonl')):
        tag = f.split('_q38_')[-1].replace('.jsonl', '')
        ds, b = NAMES.get(tag[:2]), tag[2:]
        if ds is None: continue
        by, tok = {}, {}
        for ln in open(f):
            r = json.loads(ln); c = r['config']
            by.setdefault(c, {})[r['task_id']] = bool(r['success'])
            tok.setdefault(c, {})[r['task_id']] = r['total_used']
        for arm in (UNC, FMT, OURS):
            if arm in by:
                n = len(by[arm])
                cell[(arm, b, ds)] = (sum(by[arm].values())/n*100,
                                      sum(tok[arm].values())/n)
    label = {UNC: r'\unc{}', FMT: r'\uf{}', OURS: r'\ours{}'}
    best = {}
    for b in ('8', '16', '24'):
        for ds in NAMES.values():
            vals = [(cell[(a, b, ds)][0], a) for a in (UNC, FMT, OURS) if (a, b, ds) in cell]
            if vals: best[(b, ds)] = max(vals)[1]
    out = []
    for arm in (UNC, FMT, OURS):
        for i, b in enumerate(('8', '16', '24')):
            head = f'{label[arm]} & ${b}$k ' if i == 0 else f'        & ${b}$k '
            cells = []
            for ds in ('MATH', 'GAIA', 'FRAMES', 'MuSiQue'):
                if (arm, b, ds) not in cell:
                    cells.append('--'); continue
                sr, tk = cell[(arm, b, ds)]
                s = f'{sr:.1f} / {tk:,.0f}'.replace(',', '{,}')
                cells.append(rf'\textbf{{{s}}}' if best.get((b, ds)) == arm else s)
            out.append(head + '& ' + ' & '.join(cells) + r' \\')
        out.append(r'\midrule' if arm != OURS else '')
    print('\n'.join(out))


if __name__ == '__main__' and '--latex' in sys.argv:
    print(); print('=== Table 1 (LaTeX) ==='); latex()
