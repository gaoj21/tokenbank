"""SWE credit_rate tuning at 16K: find rate that fits SWE's long code context.

  SEEDS=42,43,44 HF_TOKEN=hf_xxx .venv/bin/python -u _swe_credit_rate_sweep.py
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

BUDGET = 16000
CREDIT_RATES = [0.5, 0.75, 1.0, 1.5]
SEEDS = [int(s) for s in os.getenv('SEEDS', '42,43,44').split(',')]

TASKS, RUNNER = SWE_TASKS, run_swe_task_v2
print(f'SWE credit_rate sweep at {BUDGET}: {len(TASKS)} tasks x {len(CREDIT_RATES)} rates x {len(SEEDS)} seeds')

sweep = []
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    for cr in CREDIT_RATES:
        cfg = {
            'name': f'TB-v2-Full-cr{cr}',
            'budget': BUDGET,
            'min_quota': BUDGET // 5,
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
        print(f"\n--- TB-v2-Full @ budget={BUDGET} credit_rate={cr} seed={seed} ---")
        r = run_experiment_v2(TASKS, RUNNER, cfg, llm)
        sweep.append({'seed': seed, 'budget': BUDGET, 'credit_rate': cr,
                      'success_rate': r['success_rate'], 'cps': r['cost_per_success'],
                      'mean_tokens': r['mean_tokens'], 'cost': r['total_cost_usd'],
                      'mean_borrows': r['mean_borrows'], 'recovery_rate': r['recovery_rate']})
        print(f"  SR={r['success_rate']:.1%} CPS=${r['cost_per_success']:.4f} tokens={r['mean_tokens']:.0f} "
              f"borrows={r['mean_borrows']:.2f} recovery={r['recovery_rate']:.1%}")

rows = []
for cr in CREDIT_RATES:
    sel = [s for s in sweep if s['credit_rate'] == cr]
    rows.append({'budget': BUDGET, 'credit_rate': cr,
                 'sr_mean': float(np.mean([s['success_rate'] for s in sel])),
                 'sr_std': float(np.std([s['success_rate'] for s in sel])),
                 'cps_mean': float(np.mean([s['cps'] for s in sel])),
                 'tokens_mean': float(np.mean([s['mean_tokens'] for s in sel])),
                 'borrows_mean': float(np.mean([s['mean_borrows'] for s in sel])),
                 'recovery_mean': float(np.mean([s['recovery_rate'] for s in sel]))})
    r = rows[-1]
    print(f"cr={cr:<4} SR={r['sr_mean']:.1%}±{r['sr_std']:.1%} "
          f"CPS=${r['cps_mean']:.4f} tokens={r['tokens_mean']:.0f} "
          f"borrows={r['borrows_mean']:.2f} recovery={r['recovery_mean']:.1%}")

with open('swe_credit_rate_sweep.json', 'w') as f:
    json.dump({'seeds': SEEDS, 'raw': sweep, 'agg': rows}, f, indent=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
x = [r['credit_rate'] for r in rows]
ax1.errorbar(x, [r['sr_mean'] * 100 for r in rows], yerr=[r['sr_std'] * 100 for r in rows],
             marker='o', linewidth=2.5, capsize=5, color='#22c55e')
ax1.set_xlabel('credit_rate')
ax1.set_ylabel('Success Rate (%)')
ax1.set_title(f'SWE 16K SR vs credit_rate')
ax1.grid(True, alpha=0.3)

ax2.plot(x, [r['cps_mean'] for r in rows], marker='s', linewidth=2.5, color='#ef4444')
ax2.set_xlabel('credit_rate')
ax2.set_ylabel('CPS (USD)')
ax2.set_title(f'SWE 16K CPS vs credit_rate')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('swe_credit_rate_sweep.png', dpi=200, bbox_inches='tight')
print('\nExported: swe_credit_rate_sweep.json, swe_credit_rate_sweep.png')
