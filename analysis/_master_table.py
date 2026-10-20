#!/usr/bin/env python
"""One table with every arm, benchmark and budget measured so far.

Reads traces/ directly each time. Cells not yet run print as blank rather than
being silently dropped, so what is missing is visible.
"""
import json, os, math, sys

NM = {'ma': 'MATH', 'ga': 'GAIA', 'fr': 'FRAMES', 'mu': 'MuSiQue'}
ARMS = [('V5-Unconstrained', 'Naive'),
        ('V5-Unc-Fmt',       'Naive+Fmt'),
        ('Compaction',       'Compact'),
        ('AccumOnly',        'Accum'),
        ('V5.1-NoPool',      'Ours-v1'),
        ('V5.1-Ab7-NoVerbal','Ours')]


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); ln2 = n * math.log(2)
    ls = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-ln2 for i in range(k+1)]
    m = max(ls)
    return min(1.0, 2*math.exp(m + math.log(sum(math.exp(l-m) for l in ls))))


def cell(ds, b):
    D = {}
    def add(path, force=None):
        if not os.path.exists(path): return
        for ln in open(path):
            r = json.loads(ln)
            D.setdefault(force or r['config'], {})[r['task_id']] = r
    add(f'traces/tasks_sweep_Qwen3.8-27B_q38_{ds}{b}.jsonl')
    add(f'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{ds}{b}.jsonl')
    if b == '24':
        add(f'traces/tasks_sweep_Qwen3.8-27B_q38_ab7_{ds}.jsonl')
    add(f'traces/tasks_sweep_Qwen3.8-27B_cmp_{ds}{b}k.jsonl', 'Compaction')
    add(f'traces/tasks_sweep_Qwen3.8-27B_acc_{ds}{b}k.jsonl', 'AccumOnly')
    return D


def truncs(D, ids):
    return sum(1 for i in ids
               if D[i].get('stopped_by') == 'budget' and D[i]['total_used'] > 0.5*D[i]['budget'])


hdr = f'{"":10}{"":5}' + ''.join(f'{lab:>12}' for _, lab in ARMS)
print('SR% / mean tokens / real truncations,  vs Naive+Fmt paired McNemar')
print(hdr); print('-' * len(hdr))
for ds in ('ma', 'ga', 'fr', 'mu'):
    for b in ('8', '16', '24'):
        D = cell(ds, b)
        if 'V5-Unc-Fmt' not in D: continue
        ids = sorted(set.intersection(*(set(v) for v in D.values())))
        if not ids: continue
        F = D['V5-Unc-Fmt']
        row_sr, row_tk, row_tr = [], [], []
        for key, _ in ARMS:
            if key not in D:
                row_sr.append(f'{"":>12}'); row_tk.append(f'{"":>12}'); row_tr.append(f'{"":>12}')
                continue
            A = D[key]
            sr = sum(A[i]['success'] for i in ids)/len(ids)*100
            tk = sum(A[i]['total_used'] for i in ids)/len(ids)
            if key == 'V5-Unc-Fmt':
                mark = ''
            else:
                cw = sum(1 for i in ids if F[i]['success'] and not A[i]['success'])
                wc = sum(1 for i in ids if not F[i]['success'] and A[i]['success'])
                p = mcnemar(cw, wc)
                mark = '*' if p < 0.05 else ('.' if p < 0.10 else '')
            row_sr.append(f'{sr:>10.1f}{mark:<2}')
            row_tk.append(f'{tk:>12,.0f}')
            row_tr.append(f'{truncs(A, ids):>12}')
        print(f'{NM[ds]:10}{b+"k":>5}' + ''.join(row_sr))
        print(f'{"":10}{f"n={len(ids)}":>5}' + ''.join(row_tk))
        print(f'{"":15}' + ''.join(row_tr))
        print()
print('rows per cell: SR% / tokens / truncations.  * p<0.05, . p<0.10 vs Naive+Fmt')
print('blank = not run yet')
