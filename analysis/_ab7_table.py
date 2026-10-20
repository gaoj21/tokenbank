#!/usr/bin/env python
"""Main table with the verbalized-escalation channel removed (Ab7 as the system).

Reads traces/ every time rather than any cached summary. Two hazards this avoids,
both of which produced wrong numbers earlier in this project: trace files from
different runs can contain arms with the SAME config name, so each file is read
separately; and `stopped_by == 'budget'` also catches API timeouts that spent
almost nothing, so a real truncation additionally requires >50% of the budget.
"""
import json, math, os, sys, collections

NAMES = {'ma': 'MATH', 'ga': 'GAIA', 'fr': 'FRAMES', 'mu': 'MuSiQue'}
UNC, FMT, OURS, AB7 = ('V5-Unconstrained', 'V5-Unc-Fmt',
                       'V5.1-NoPool', 'V5.1-Ab7-NoVerbal')


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); ln2 = n * math.log(2)
    ls = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-ln2 for i in range(k+1)]
    m = max(ls)
    return min(1.0, 2*math.exp(m + math.log(sum(math.exp(l-m) for l in ls))))


def load(path):
    d = {}
    if not os.path.exists(path): return d
    for ln in open(path):
        r = json.loads(ln)
        d.setdefault(r['config'], {})[r['task_id']] = r
    return d


def cells():
    out = {}
    for ds in NAMES:
        for b in ('8', '16', '24'):
            base = load(f'traces/tasks_sweep_Qwen3.8-27B_q38_{ds}{b}.jsonl')
            ab = load(f'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{ds}{b}.jsonl')
            if not ab and b == '24':
                ab = load(f'traces/tasks_sweep_Qwen3.8-27B_q38_ab7_{ds}.jsonl')
            if AB7 not in ab or FMT not in base: continue
            arms = {UNC: base.get(UNC, {}), FMT: base[FMT],
                    OURS: base.get(OURS, {}), AB7: ab[AB7]}
            ids = sorted(set(arms[AB7]) & set(arms[FMT]))
            out[(NAMES[ds], b)] = (arms, ids)
    return out


def trunc(D, ids):
    return sum(1 for i in ids if D[i].get('stopped_by') == 'budget'
               and D[i]['total_used'] > 0.5 * D[i]['budget'])


C = cells()
print(f'{"set":9}{"B":>5}{"n":>6}{"Naive":>8}{"+Fmt":>8}{"Ours":>8}{"Ab7":>8}'
      f'{"Ab7-Fmt":>9}{"W/L":>9}{"p":>10}{"tok":>7}{"trunc":>10}')
print('-' * 97)
for ds in ('MATH', 'GAIA', 'FRAMES', 'MuSiQue'):
    for b in ('8', '16', '24'):
        if (ds, b) not in C: continue
        arms, ids = C[(ds, b)]
        A, F = arms[AB7], arms[FMT]
        sr = lambda D: sum(D[i]['success'] for i in ids)/len(ids)*100
        tk = lambda D: sum(D[i]['total_used'] for i in ids)/len(ids)
        cw = sum(1 for i in ids if F[i]['success'] and not A[i]['success'])
        wc = sum(1 for i in ids if not F[i]['success'] and A[i]['success'])
        o = arms[OURS]; u = arms[UNC]
        print(f'{ds:9}{b+"k":>5}{len(ids):6}'
              f'{sr(u) if u else float("nan"):7.1f}%{sr(F):7.1f}%'
              f'{sr(o) if o else float("nan"):7.1f}%{sr(A):7.1f}%'
              f'{sr(A)-sr(F):+9.1f}{f"{wc}/{cw}":>9}{mcnemar(cw,wc):10.1e}'
              f'{tk(A)/tk(F)*100-100:+6.0f}%{f"{trunc(F,ids)}->{trunc(A,ids)}":>10}')

if '--latex' in sys.argv:
    print('\n=== Table 1 (LaTeX) ===')
    lab = {UNC: r'\unc{}', FMT: r'\uf{}', AB7: r'\ours{}'}
    best = {}
    for (ds, b), (arms, ids) in C.items():
        vals = [(sum(arms[a][i]['success'] for i in ids)/len(ids), a)
                for a in (UNC, FMT, AB7) if arms[a]]
        best[(ds, b)] = max(vals)[1]
    for arm in (UNC, FMT, AB7):
        for i, b in enumerate(('8', '16', '24')):
            head = f'{lab[arm]} & ${b}$k ' if i == 0 else f'        & ${b}$k '
            cs = []
            for ds in ('MATH', 'GAIA', 'FRAMES', 'MuSiQue'):
                if (ds, b) not in C or not C[(ds, b)][0][arm]:
                    cs.append('--'); continue
                arms, ids = C[(ds, b)]
                D = arms[arm]
                s = (f'{sum(D[i]["success"] for i in ids)/len(ids)*100:.1f} / '
                     f'{sum(D[i]["total_used"] for i in ids)/len(ids):,.0f}').replace(',', '{,}')
                cs.append(rf'\textbf{{{s}}}' if best[(ds, b)] == arm else s)
            print(head + '& ' + ' & '.join(cs) + r' \\')
        print(r'\midrule' if arm != AB7 else '')
