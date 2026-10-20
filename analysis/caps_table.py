"""Appendix table: accuracy change of the plain pipeline from a 1,024 to a 4,096 per-call cap, by
task budget (MATH and GAIA, both models), paired exact McNemar. Uses re-scored MATH traces."""
import json, statistics as st
from scipy.stats import binomtest
load = lambda p: {d['task_id']: d for d in (json.loads(l) for l in open(p)) if d['config'] == 'V5-Unc-Fmt'}
M = {'GLM': ('GLM-5.3-Flash', 'glm4', {'24': 'glmx4k', '16': 'glmr4k', '8': 'glmr4k'}),
     'DS': ('DeepSeek-V4.1-Flash', 'ds', {'24': 'dsx4k', '16': 'dsr4k', '8': 'dsr4k'})}
out = []
for fam, (T, f1, f4) in M.items():
    for t, ds in (('ma', 'MATH'), ('ga', 'GAIA')):
        row = dict(cell=f'{fam} {ds}')
        for k in ('24', '16', '8'):
            a = load(f'traces/tasks_sweep_{T}_{f1}_{t}{k}.jsonl'); b = load(f'traces/tasks_sweep_{T}_{f4[k]}_{t}{k}.jsonl')
            ids = [i for i in a if i in b]
            d = 100 * (st.mean(b[i]['success'] for i in ids) - st.mean(a[i]['success'] for i in ids))
            x = sum(1 for i in ids if b[i]['success'] and not a[i]['success']); y = sum(1 for i in ids if a[i]['success'] and not b[i]['success'])
            p = binomtest(min(x, y), x + y, .5).pvalue if x + y else 1
            row[k] = (round(d, 1), bool(p < 0.05))
        out.append(row); print(row)
json.dump(out, open('results/caps_table.json', 'w'), indent=1)
