"""Which Gemma bf16 cells are actually complete.

The chain's own skip test is `trace file exists and is non-empty`, which is
wrong: the trace is appended while the cell runs, so a cell interrupted halfway
looks finished. A cell is complete only when it holds n_tasks x 3 arms rows.
Check with this before any restart; move short cells aside so they re-run.
"""
import json, os, sys, collections
N = {'ga': 122, 'ma': 400, 'fr': 660, 'mu': 1934}
ARMS = 3
rows = []
for ds in ['ga', 'ma', 'fr', 'mu']:
    for b in ['24', '16', '8']:
        f = f'tasks_sweep_gemma-4-31B-bf16_gor_{ds}{b}.jsonl'
        want = N[ds] * ARMS
        if not os.path.exists(f):
            rows.append((f'{ds}{b}', 0, want, '未开始')); continue
        seen = collections.Counter()
        for l in open(f):
            r = json.loads(l); seen[r['config']] += 1
        got = sum(seen.values())
        state = '完成' if got >= want else f'不完整 {dict(seen)}'
        rows.append((f'{ds}{b}', got, want, state))
bad = [r for r in rows if 0 < r[1] < r[2]]
for name, got, want, state in rows:
    print(f'  {name:5} {got:5}/{want:<5} {state}')
if bad:
    print('\n  重启前先移走这些不完整的格:')
    for name, *_ in bad:
        print(f'    mv tasks_sweep_gemma-4-31B-bf16_gor_{name}.jsonl _aborted/')
