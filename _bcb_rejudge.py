#!/usr/bin/env python
"""Re-score every BigCodeBench run offline, from the stored replies.

The inline judgements are unusable. The judge container was given 2 CPUs while
64 sweep workers issued concurrent `docker exec` judgements into it, so tests
hit the 30s wall clock through contention and were recorded as failures. The
signature was an arm whose token usage does not change with budget (Ab7 never
escalates: 1192/1195/1197 tokens at 4k/8k/16k) nevertheless scoring 27.2%,
33.0% and 20.8%. Re-judged without contention the same replies give ~45% at
every budget.

DUMP_RAW_ANSWER kept the full reply for every task, so nothing needs re-running.
Writes `success` back into a parallel trace so the originals stay intact.
"""
import os, sys, json, glob, collections
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bcb_judge import judge_bigcodebench
from datasets import load_dataset

W = int(os.getenv('REJUDGE_WORKERS', '6'))
tests = {f"bcb_{it['task_id']}": it['test']
         for it in load_dataset('bigcode/bigcodebench', split='v0.1.4')}

# The glob is a parameter: the same replies need re-judging for every model,
# and the Gemma traces land in the working directory rather than traces/.
PATTERN = os.getenv('REJUDGE_GLOB', 'traces/tasks_sweep_Qwen3.8-27B_bcb*.jsonl')
for f in sorted(glob.glob(PATTERN)):
    rows = [json.loads(l) for l in open(f)]
    out = f.replace('.jsonl', '_rejudged.jsonl')
    if os.path.exists(out):
        print(f'{os.path.basename(f)}: already re-judged, skipping')
        continue
    def one(r):
        ok = judge_bigcodebench(r.get('raw_answer') or '', tests.get(r['task_id'], ''))
        r = dict(r); r['success'] = bool(ok); r['f1'] = 1.0 if ok else 0.0
        return r
    with cf.ThreadPoolExecutor(max_workers=W) as ex:
        new = list(ex.map(one, rows))
    with open(out, 'w') as fh:
        for r in new:
            fh.write(json.dumps(r) + '\n')
    before = collections.Counter()
    after = collections.Counter()
    n = collections.Counter()
    for a, b in zip(rows, new):
        n[a['config']] += 1
        before[a['config']] += a['success']
        after[b['config']] += b['success']
    tag = os.path.basename(f).split('_bcb')[-1].replace('.jsonl', '')
    print(f'\n{tag}:')
    for c in n:
        print(f'  {c:12} inline {before[c]/n[c]*100:5.1f}%  ->  offline {after[c]/n[c]*100:5.1f}%'
              f'   ({before[c]} -> {after[c]} of {n[c]})')
