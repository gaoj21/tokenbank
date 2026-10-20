"""32K multi-seed confirmation: Unconstrained / Static Quota / TB-v2-Full.

  SEEDS=42,43,44 HF_TOKEN=hf_xxx .venv/bin/python -u _full_confirm_32k.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re as _re
src = open('_run_v2_experiment.py').read()
_parts = _re.split(r'# ============ NOTEBOOK CELL (\d+) ============\n', src)
cell_map = {int(_parts[i]): _parts[i + 1] for i in range(1, len(_parts), 2)}

for n in [2, 4, 6, 7, 8, 9, 10]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())

BUDGET = 32000
SEEDS = [int(s) for s in os.getenv('SEEDS', '42,43,44').split(',')]

CONFIGS = [
    {'name': 'Unconstrained', 'budget': 200000, 'policy': 'hard', 'external_plan': None,
     'gamma': 0.0, 'enable_predictive': False, 'enable_borrowing': False, 'enable_opl': False, 'enable_kv': False, 'theta': 1.0},
    {'name': 'Static Quota', 'policy': 'hard', 'external_plan': None,
     'gamma': 1.0, 'enable_predictive': False, 'enable_borrowing': False, 'enable_opl': False, 'enable_kv': False, 'theta': 1.0},
    {'name': 'TB-v2-Full', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True, 'theta': 0.8},
]

TASKS, RUNNER = GAIA_TASKS, run_gaia_task_v2
print(f'32K confirmation on GAIA: {len(TASKS)} tasks x {len(CONFIGS)} configs x {len(SEEDS)} seeds')

sweep = []
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    for cfg in CONFIGS:
        c = dict(cfg)
        if c.get('budget') != 200000:
            c['budget'] = BUDGET
            c['min_quota'] = BUDGET // 5
        print(f"\n--- {c['name']} @ budget={c['budget']} seed={seed} ---")
        r = run_experiment_v2(TASKS, RUNNER, c, llm)
        sweep.append({'seed': seed, 'budget': c['budget'], 'name': c['name'],
                      'success_rate': r['success_rate'], 'cps': r['cost_per_success'],
                      'mean_tokens': r['mean_tokens'], 'cost': r['total_cost_usd'],
                      'mean_borrows': r['mean_borrows'], 'recovery_rate': r['recovery_rate']})
        print(f"  SR={r['success_rate']:.1%} CPS=${r['cost_per_success']:.4f} tokens={r['mean_tokens']:.0f} "
              f"borrows={r['mean_borrows']:.2f} recovery={r['recovery_rate']:.1%}")

rows = []
for name in [c['name'] for c in CONFIGS]:
    sel = [s for s in sweep if s['name'] == name]
    rows.append({'budget': BUDGET, 'name': name,
                 'sr_mean': float(np.mean([s['success_rate'] for s in sel])),
                 'sr_std': float(np.std([s['success_rate'] for s in sel])),
                 'cps_mean': float(np.mean([s['cps'] for s in sel])),
                 'tokens_mean': float(np.mean([s['mean_tokens'] for s in sel])),
                 'borrows_mean': float(np.mean([s['mean_borrows'] for s in sel])),
                 'recovery_mean': float(np.mean([s['recovery_rate'] for s in sel]))})
    r = rows[-1]
    print(f"budget={BUDGET:>6} {name:>14} SR={r['sr_mean']:.1%}±{r['sr_std']:.1%} "
          f"CPS=${r['cps_mean']:.4f} tokens={r['tokens_mean']:.0f} "
          f"borrows={r['borrows_mean']:.2f} recovery={r['recovery_mean']:.1%}")

with open('full_confirm_32k.json', 'w') as f:
    json.dump({'seeds': SEEDS, 'raw': sweep, 'agg': rows}, f, indent=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = {'Unconstrained': '#ef4444', 'Static Quota': '#f97316', 'TB-v2-Full': '#22c55e'}
x = list(range(len(rows)))
labels = [r['name'] for r in rows]
ax1.bar(x, [r['sr_mean'] * 100 for r in rows], yerr=[r['sr_std'] * 100 for r in rows],
        capsize=5, color=[colors.get(l, '#3b82f6') for l in labels])
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha='right')
ax1.set_ylabel('Success Rate (%)')
ax1.set_title(f'32K Success Rate (GAIA, n={len(TASKS)} x {len(SEEDS)} seeds)')
ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(x, [r['cps_mean'] for r in rows], color=[colors.get(l, '#3b82f6') for l in labels])
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha='right')
ax2.set_ylabel('CPS (USD)')
ax2.set_title(f'32K Cost per Success (GAIA)')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('full_confirm_32k.png', dpi=200, bbox_inches='tight')
print('\nExported: full_confirm_32k.json, full_confirm_32k.png')
