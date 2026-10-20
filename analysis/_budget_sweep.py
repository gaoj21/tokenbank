"""Budget sweep: find the budget regime where TB-v2 dominates.

Reuses dataset + classes + runners from _run_v2_experiment.py (cells 2-10),
then runs {Unconstrained, Static Quota, TB-v2-Full} x BUDGETS on GAIA.

  SEEDS=42 HF_TOKEN=hf_xxx .venv/bin/python -u _budget_sweep.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- load definitions from the experiment script (cells 2 through 10) ----
import re as _re
src = open('_run_v2_experiment.py').read()
_parts = _re.split(r'# ============ NOTEBOOK CELL (\d+) ============\n', src)
cell_map = {int(_parts[i]): _parts[i + 1] for i in range(1, len(_parts), 2)}

for n in [2, 4, 6, 7, 8, 9, 10]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())

BUDGETS_SWEEP = [32000, 48000]
SEEDS = [int(s) for s in os.getenv('SEEDS', '42').split(',')]

CONFIGS = [
    {'name': 'Unconstrained', 'budget': 200000, 'policy': 'hard', 'external_plan': None,
     'gamma': 0.0, 'enable_predictive': False, 'enable_borrowing': False, 'enable_opl': False, 'enable_kv': False, 'theta': 1.0},
    {'name': 'Static Quota', 'policy': 'hard', 'external_plan': None,
     'gamma': 1.0, 'enable_predictive': False, 'enable_borrowing': False, 'enable_opl': False, 'enable_kv': False, 'theta': 1.0},
    {'name': 'TB-v2-Full', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True, 'theta': 0.8},
    {'name': 'TB-v2-Scored', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True, 'theta': 0.8,
     'enable_credit_scoring': True},
]

TASKS, RUNNER = GAIA_TASKS, run_gaia_task_v2
print(f'Budget sweep on GAIA: {len(TASKS)} tasks x {len(CONFIGS)} configs x {len(BUDGETS_SWEEP)} budgets x {len(SEEDS)} seeds')

sweep = []
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    for budget in BUDGETS_SWEEP:
        for cfg in CONFIGS:
            c = dict(cfg)
            if c.get('budget') != 200000:
                c['budget'] = budget
                c['min_quota'] = budget // 5
            print(f"\n--- {c['name']} @ budget={c['budget']} seed={seed} ---")
            r = run_experiment_v2(TASKS, RUNNER, c, llm)
            sweep.append({'seed': seed, 'budget': budget, 'name': c['name'],
                          'success_rate': r['success_rate'], 'cps': r['cost_per_success'],
                          'mean_tokens': r['mean_tokens'], 'cost': r['total_cost_usd']})
            print(f"  SR={r['success_rate']:.1%} CPS=${r['cost_per_success']:.4f} tokens={r['mean_tokens']:.0f}")

# aggregate over seeds
rows = []
for budget in BUDGETS_SWEEP:
    for name in [c['name'] for c in CONFIGS]:
        sel = [s for s in sweep if s['budget'] == budget and s['name'] == name]
        rows.append({'budget': budget, 'name': name,
                     'sr_mean': float(np.mean([s['success_rate'] for s in sel])),
                     'sr_std': float(np.std([s['success_rate'] for s in sel])),
                     'cps_mean': float(np.mean([s['cps'] for s in sel])),
                     'tokens_mean': float(np.mean([s['mean_tokens'] for s in sel]))})
        r = rows[-1]
        print(f"budget={budget:>6} {name:>14} SR={r['sr_mean']:.1%}±{r['sr_std']:.1%} CPS=${r['cps_mean']:.4f} tokens={r['tokens_mean']:.0f}")

with open('budget_sweep_confirm.json', 'w') as f:
    json.dump({'seeds': SEEDS, 'raw': sweep, 'agg': rows}, f, indent=2)

# plot SR and CPS vs budget
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = {'Unconstrained': '#ef4444', 'Static Quota': '#f97316', 'TB-v2-Full': '#22c55e', 'TB-v2-Scored': '#3b82f6'}
for name, color in colors.items():
    sel = [r for r in rows if r['name'] == name]
    x = [r['budget'] for r in sel]
    ax1.errorbar(x, [r['sr_mean'] * 100 for r in sel], yerr=[r['sr_std'] * 100 for r in sel],
                 marker='o', linewidth=2.5, capsize=5, label=name, color=color)
    ax2.plot(x, [r['cps_mean'] for r in sel], marker='s', linewidth=2.5, label=name, color=color)
for ax, title, ylabel in [(ax1, 'Success Rate vs Budget (GAIA)', 'Success Rate (%)'),
                          (ax2, 'Cost per Success vs Budget (GAIA)', 'CPS (USD)')]:
    ax.set_xscale('log')
    ax.set_xlabel('Per-task budget (tokens)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
ax2.set_yscale('log')
plt.tight_layout()
plt.savefig('budget_sweep_confirm.png', dpi=200, bbox_inches='tight')
print('\nExported: budget_sweep_confirm.json, budget_sweep_confirm.png')
