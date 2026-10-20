"""TB-v2-Full baseline at 8K with credit_rate=0.5, 3 seeds.

  SEEDS=42,43,44 HF_TOKEN=hf_xxx .venv/bin/python -u _full_baseline_8k.py
"""
import os
import json
import numpy as np

import re as _re
src = open('_run_v2_experiment.py').read()
_parts = _re.split(r'# ============ NOTEBOOK CELL (\d+) ============\n', src)
cell_map = {int(_parts[i]): _parts[i + 1] for i in range(1, len(_parts), 2)}

for n in [2, 4, 6, 7, 8, 9, 10]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())

BUDGET = 8000
SEEDS = [int(s) for s in os.getenv('SEEDS', '42,43,44').split(',')]

TASKS, RUNNER = GAIA_TASKS, run_gaia_task_v2
print(f'Full baseline on GAIA (8K, credit_rate=0.5): {len(TASKS)} tasks x {len(SEEDS)} seeds')

results = []
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    cfg = {
        'name': 'TB-v2-Full',
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
        'credit_rate': 0.5,
    }
    print(f"\n--- seed={seed} ---")
    r = run_experiment_v2(TASKS, RUNNER, cfg, llm)
    results.append({
        'seed': seed,
        'success_rate': r['success_rate'],
        'cps': r['cost_per_success'],
        'mean_tokens': r['mean_tokens'],
        'cost': r['total_cost_usd'],
        'mean_borrows': r['mean_borrows'],
        'recovery_rate': r['recovery_rate']
    })
    print(f"  SR={r['success_rate']:.1%} CPS=${r['cost_per_success']:.4f} tokens={r['mean_tokens']:.0f} "
          f"borrows={r['mean_borrows']:.2f} recovery={r['recovery_rate']:.1%}")

agg = {
    'budget': BUDGET,
    'credit_rate': 0.5,
    'sr_mean': float(np.mean([r['success_rate'] for r in results])),
    'sr_std': float(np.std([r['success_rate'] for r in results])),
    'cps_mean': float(np.mean([r['cps'] for r in results])),
    'tokens_mean': float(np.mean([r['mean_tokens'] for r in results])),
    'borrows_mean': float(np.mean([r['mean_borrows'] for r in results])),
    'recovery_mean': float(np.mean([r['recovery_rate'] for r in results]))
}
print(f"\n=== Aggregate ===")
print(f"SR={agg['sr_mean']:.1%}±{agg['sr_std']:.1%} CPS=${agg['cps_mean']:.4f} "
      f"tokens={agg['tokens_mean']:.0f} borrows={agg['borrows_mean']:.2f} recovery={agg['recovery_mean']:.1%}")

with open('full_baseline_8k.json', 'w') as f:
    json.dump({'seeds': SEEDS, 'raw': results, 'agg': agg}, f, indent=2)
print('\nExported: full_baseline_8k.json')
