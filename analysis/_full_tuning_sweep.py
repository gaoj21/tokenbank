"""TB-v2-Full tuning sweep: credit_rate at 8K/16K.

  SEEDS=42 HF_TOKEN=hf_xxx .venv/bin/python -u _full_tuning_sweep.py
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

BUDGETS_SWEEP = [8000, 16000]
CREDIT_RATES = [0.5, 0.75, 1.0, 1.5]
SEEDS = [int(s) for s in os.getenv('SEEDS', '42').split(',')]

TASKS, RUNNER = GAIA_TASKS, run_gaia_task_v2
print(f'Full tuning sweep on GAIA: {len(TASKS)} tasks x {len(CREDIT_RATES)} credit_rates x {len(BUDGETS_SWEEP)} budgets x {len(SEEDS)} seeds')

sweep = []
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    for budget in BUDGETS_SWEEP:
        for cr in CREDIT_RATES:
            cfg = {
                'name': f'TB-v2-Full-cr{cr}',
                'budget': budget,
                'min_quota': budget // 5,
                'policy': 'hard',
                'external_plan': None,
                'gamma': 0.8,
                'enable_predictive': True,
                'enable_borrowing': True,
                'enable_opl': True,
                'enable_kv': True,
                'theta': 0.8,
                'credit_rate': cr,
            }
            print(f"\n--- TB-v2-Full @ budget={budget} credit_rate={cr} seed={seed} ---")
            r = run_experiment_v2(TASKS, RUNNER, cfg, llm)
            sweep.append({
                'seed': seed, 'budget': budget, 'credit_rate': cr,
                'success_rate': r['success_rate'], 'cps': r['cost_per_success'],
                'mean_tokens': r['mean_tokens'], 'cost': r['total_cost_usd'],
                'mean_borrows': r['mean_borrows'], 'recovery_rate': r['recovery_rate']
            })
            print(f"  SR={r['success_rate']:.1%} CPS=${r['cost_per_success']:.4f} tokens={r['mean_tokens']:.0f} "
                  f"borrows={r['mean_borrows']:.2f} recovery={r['recovery_rate']:.1%}")

rows = []
for budget in BUDGETS_SWEEP:
    for cr in CREDIT_RATES:
        sel = [s for s in sweep if s['budget'] == budget and s['credit_rate'] == cr]
        rows.append({
            'budget': budget, 'credit_rate': cr,
            'sr_mean': float(np.mean([s['success_rate'] for s in sel])),
            'sr_std': float(np.std([s['success_rate'] for s in sel])),
            'cps_mean': float(np.mean([s['cps'] for s in sel])),
            'tokens_mean': float(np.mean([s['mean_tokens'] for s in sel])),
            'borrows_mean': float(np.mean([s['mean_borrows'] for s in sel])),
            'recovery_mean': float(np.mean([s['recovery_rate'] for s in sel]))
        })
        r = rows[-1]
        print(f"budget={budget:>6} cr={cr:<4} SR={r['sr_mean']:.1%}±{r['sr_std']:.1%} "
              f"CPS=${r['cps_mean']:.4f} tokens={r['tokens_mean']:.0f} "
              f"borrows={r['borrows_mean']:.2f} recovery={r['recovery_mean']:.1%}")

with open('full_credit_rate_sweep.json', 'w') as f:
    json.dump({'seeds': SEEDS, 'raw': sweep, 'agg': rows}, f, indent=2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
colors = {8000: '#ef4444', 16000: '#22c55e'}
for budget in BUDGETS_SWEEP:
    sel = [r for r in rows if r['budget'] == budget]
    x = [r['credit_rate'] for r in sel]
    ax1.plot(x, [r['sr_mean'] * 100 for r in sel], marker='o', linewidth=2.5, label=f'{budget}K', color=colors[budget])
    ax2.plot(x, [r['cps_mean'] for r in sel], marker='s', linewidth=2.5, label=f'{budget}K', color=colors[budget])
    ax3.plot(x, [r['borrows_mean'] for r in sel], marker='^', linewidth=2.5, label=f'{budget}K', color=colors[budget])

ax1.set_xlabel('credit_rate')
ax1.set_ylabel('Success Rate (%)')
ax1.set_title('SR vs credit_rate')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_xlabel('credit_rate')
ax2.set_ylabel('CPS (USD)')
ax2.set_title('CPS vs credit_rate')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend()

ax3.set_xlabel('credit_rate')
ax3.set_ylabel('Borrows / task')
ax3.set_title('Borrowing vs credit_rate')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig('full_credit_rate_sweep.png', dpi=200, bbox_inches='tight')
print('\nExported: full_credit_rate_sweep.json, full_credit_rate_sweep.png')
