"""State-compression ablation (GLM, 24k budget, 4,096 cap): accuracy change of each arm from the
plain pipeline (V5-Unc-Fmt, glmx4k) on the same tasks, exact McNemar. Uses re-scored MATH traces."""
import json, statistics as st
from scipy.stats import binomtest
def load(path, cfg):
    return {d['task_id']: d for d in (json.loads(l) for l in open(path)) if d['config'] == cfg}
ARMS = [('Plan → current step', 'glmab', 'ST-PlanOnly'), ('Tool output → 120-token note', 'glmab', 'ST-ToolOnly'),
        ('Full history (accumulate)', 'glmst', 'ST-Accum'), ('Structured critique, cap 150', 'glmrv', 'ST-CritOnly'),
        ('Structured critique, cap 1,024', 'glmrv', 'ST-CritOnly-R1k')]
out = {'settings': ['GLM MATH', 'GLM GAIA'], 'rows': []}
for name, tag, cfg in ARMS:
    vals, sig = [], []
    for ab in ('ma', 'ga'):
        base = load(f'traces/tasks_sweep_GLM-5.3-Flash_glmx4k_{ab}24.jsonl', 'V5-Unc-Fmt')
        arm = load(f'traces/tasks_sweep_GLM-5.3-Flash_{tag}_{ab}24.jsonl', cfg)
        ids = [i for i in arm if i in base]
        d = 100 * (st.mean(arm[i]['success'] for i in ids) - st.mean(base[i]['success'] for i in ids))
        b = sum(1 for i in ids if base[i]['success'] and not arm[i]['success']); c = sum(1 for i in ids if arm[i]['success'] and not base[i]['success'])
        p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        vals.append(round(d, 1)); sig.append(int(p < 0.05))
        print(f'{name:<32} {ab}  n={len(ids):<4} delta {d:+.1f}  p={p:.3f}')
    out['rows'].append([name, vals, sig])
json.dump(out, open('paper/figures/fig_state.json', 'w'), indent=1)
