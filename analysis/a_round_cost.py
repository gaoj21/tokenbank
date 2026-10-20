"""Direction A: per-round prompt cost as a function of the round index, three regimes.

Accumulate appends every round (kappa grows in t). Naive+Fmt rebuilds the context each round
but carries the full plan, the full previous answer and the full critique, so kappa is
constant in t but set by the previous round's verbosity. Governed rebuilds and caps each slot.
Only L2 rounds are compared: for the governed arm the first two records are L0 and L1.
"""
import json, os, collections, statistics as st, sys
import numpy as np

def l2_rounds(d):
    rs = [r for r in d['rounds'] if r['role'] == 'executor']
    if d['config'] in ('V5.1-Ab7-NoVerbal', 'V5.2-ReasonLadder'):
        if d.get('escalation_level', 0) < 2: return None
        rs = rs[2:]
    return rs

import sys; sys.path.insert(0,"analysis"); from _files import path
FILES_OLD = {
 'Qwen3.8-27B': {
   'Accumulate': 'traces/tasks_sweep_Qwen3.8-27B_acc_{t}{k}k.jsonl',
   'Naive+Fmt':  'traces/tasks_sweep_Qwen3.8-27B_q38_{t}{k}.jsonl',
   'Governed':   'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{t}{k}.jsonl'},
 'Gemma-4-31B': {
   'Accumulate': 'traces/tasks_sweep_gemma-4-31B-bf16_gcmp_{t}{k}.jsonl',
   'Naive+Fmt':  'traces/tasks_sweep_gemma-4-31B-bf16_gor_{t}{k}.jsonl',
   'Governed':   'traces/tasks_sweep_gemma-4-31B-bf16_gor_{t}{k}.jsonl'},
}
ARM = {'Accumulate':'V5-Accum-Only','Naive+Fmt':'V5-Unc-Fmt','Governed':'V5.1-Ab7-NoVerbal'}
out = {}
for fam in ['Qwen','Gemma']:
    for arm in ARM:
        by_t = collections.defaultdict(list)
        for t in ['ma','ga','fr','mu']:
            p = path(fam, ARM[arm], t, 24)
            if not p or not os.path.exists(p): continue
            for l in open(p):
                try: d = json.loads(l)
                except: continue
                if d['config'] != ARM[arm]: continue
                rs = l2_rounds(d)
                if not rs: continue
                for i, r in enumerate(rs[:5]):
                    by_t[i].append(r['prompt_tokens'])
        means = [st.mean(by_t[i]) for i in range(5) if len(by_t[i]) >= 20]
        ns = [len(by_t[i]) for i in range(5) if len(by_t[i]) >= 20]
        x = np.arange(len(means))
        slope = np.polyfit(x, means, 1)[0] if len(means) > 1 else float('nan')
        out[(fam, arm)] = (means, ns, slope)
        print(f'{fam:<12}{arm:<12} ' + ' '.join(f'{m:>6.0f}' for m in means)
              + f'   slope {slope:+.0f}/round   n={ns}')
json.dump({f'{a}|{b}': v for (a, b), v in out.items()}, open('results/a_round_cost.json', 'w'))
