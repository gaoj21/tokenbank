#!/usr/bin/env python
"""Qwen3.8 replacement for the paper's failure-mode table (Table 3).

Splits correct->wrong transitions into the two modes the paper distinguishes:
  revision   -- a later round overwrites an answer that was correct
  truncation -- the budget-exhaustion call replaces it
Truncation requires actually spending the budget: `stopped_by == 'budget'` also
catches API timeouts, which spend ~0 tokens and must not be counted here.
The conditional rate is flips / items ever correct at some round, which is the
comparable metric when one arm reaches a correct answer far more often.
"""
import json, glob, collections

NM = {'ma':'MATH','ga':'GAIA','fr':'FRAMES','mu':'MuSiQue'}
ARMS = ['V5-Unc-Fmt', 'V5.1-NoPool']

def scan(path):
    out = collections.defaultdict(list)
    for ln in open(path):
        r = json.loads(ln)
        ever = any(str(x.get('is_correct')) == 'True' for x in r.get('rounds', []))
        truncated = (r.get('stopped_by') == 'budget'
                     and r['total_used'] > 0.5 * r['budget'])
        out[r['config']].append((bool(r['success']), ever, truncated))
    return out

print(f'{"set":9}{"B":>5} | ' + ' | '.join(f'{a:>26}' for a in ARMS))
print(f'{"":14} | ' + ' | '.join(f'{"rev":>7}{"trunc":>7}{"cond":>7}{"n":>5}' for _ in ARMS))
print('-' * 74)
for f in sorted(glob.glob('traces/tasks_sweep_Qwen3.8-27B_q38_*.jsonl')):
    tag = f.split('_q38_')[-1].replace('.jsonl', '')
    d = scan(f)
    if not all(a in d for a in ARMS): continue
    cells = []
    for a in ARMS:
        v = d[a]
        ever = sum(e for _, e, _ in v)
        trunc = sum(1 for s, e, t in v if e and not s and t)
        rev = sum(1 for s, e, t in v if e and not s and not t)
        cells.append(f'{rev:7}{trunc:7}{(rev+trunc)/ever*100 if ever else 0:6.1f}%{ever:5}')
    print(f'{NM.get(tag[:2], tag):9}{tag[2:]+"k":>5} | ' + ' | '.join(cells))
print('-' * 74)
print('rev = revision (a later round overwrote a correct answer)')
print('trunc = budget-exhaustion replaced it; cond = (rev+trunc) / ever-correct; n = ever correct')
