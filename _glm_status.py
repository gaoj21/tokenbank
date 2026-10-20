"""Which GLM cells are complete.

A cell is done when every arm holds n_tasks rows. The trace is appended while a cell runs,
so its existence proves nothing. The status word is ASCII and the two states do not share a
substring, because an earlier version printed a Chinese pair where one contained the other
and the chain's `grep` skipped a cell that had only two of its three arms.
"""
import json, os, collections, sys
N = {'ga': 122, 'ma': 400, 'fr': 660, 'mu': 1934}
ARMS = ['V5-Unconstrained', 'V5-Unc-Fmt', 'V5.2-ReasonLadder']
for ds in ['ga', 'ma', 'fr', 'mu']:
    for b in ['24', '16', '8']:
        cell = f'{ds}{b}'
        f = f'tasks_sweep_GLM-5.3-Flash_glm4_{cell}.jsonl'
        c = collections.Counter()
        if os.path.exists(f):
            for l in open(f): c[json.loads(l)['config']] += 1
        done = all(c.get(a, 0) >= N[ds] for a in ARMS)
        have = {a.split('-')[-1]: c.get(a, 0) for a in ARMS}
        print(f'{cell:6} {"DONE" if done else "TODO"} {sum(c.values()):6} rows  {have}')
