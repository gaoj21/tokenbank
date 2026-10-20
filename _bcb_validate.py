#!/usr/bin/env python
"""Validate the BigCodeBench judge against the benchmark's own canonical solutions.

The lesson from the SWE patch builder: a self-consistency check passes happily
while the thing is wrong. The only check worth running is against an external
ground truth. Here that truth is the dataset's canonical_solution, which by
construction passes the task's tests. A canonical solution the judge marks wrong
is a judge bug (wrong assembly, missing import, too short a timeout), not a
model failure.
"""
import sys, os, json, random, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bcb_judge import judge_bigcodebench
from datasets import load_dataset

N = int(os.getenv('N', '60'))
ds = load_dataset('bigcode/bigcodebench', split='v0.1.4')
items = list(ds)
random.Random(42).shuffle(items)
items = items[:N]


def check(it):
    # canonical_solution is the body; code_prompt carries the imports and signature
    sol = (it['code_prompt'] or '') + (it['canonical_solution'] or '')
    reply = f"```python\n{sol}\n```"
    return it['task_id'], judge_bigcodebench(reply, it['test'])


with cf.ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(check, items))
ok = [t for t, v in res if v]
bad = [t for t, v in res if not v]
print(f'canonical solutions passing the judge: {len(ok)}/{len(res)}')
if bad:
    print('failures (judge bugs unless the task is known-broken):')
    for t in bad[:10]:
        print('  ', t)
json.dump({'ok': len(ok), 'n': len(res), 'failures': bad},
          open('results/bcb_judge_validation.json', 'w'))
