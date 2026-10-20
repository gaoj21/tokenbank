"""Credit scoring experiment: flat credit line vs bank-style credit scoring.

  SEEDS=42 HF_TOKEN=hf_xxx .venv/bin/python -u _credit_experiment.py
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

BUDGETS = [2000, 4000, 8000]
SEEDS = [int(s) for s in os.getenv('SEEDS', '42').split(',')]
CONFIGS = [
    {'name': 'TB-v2-FlatCredit', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True,
     'enable_opl': True, 'enable_kv': True, 'theta': 0.8, 'enable_credit_scoring': False},
    {'name': 'TB-v2-ScoredCredit', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True,
     'enable_opl': True, 'enable_kv': True, 'theta': 0.8, 'enable_credit_scoring': True},
]

TASKS, RUNNER = GAIA_TASKS, run_gaia_task_v2
print(f'Credit experiment on GAIA: {len(TASKS)} tasks x {len(CONFIGS)} configs x {len(BUDGETS)} budgets x {len(SEEDS)} seeds')

out = []
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    for budget in BUDGETS:
        for cfg in CONFIGS:
            c = dict(cfg)
            c['budget'] = budget
            c['min_quota'] = budget // 5
            print(f"\n--- {c['name']} @ budget={budget} seed={seed} ---")
            r = run_experiment_v2(TASKS, RUNNER, c, llm)
            borrowed_tokens = sum(t['bank_summary']['total_borrowed'] for t in r['results'])
            successes = sum(t['success'] for t in r['results'])
            row = {'seed': seed, 'budget': budget, 'name': c['name'],
                   'success_rate': r['success_rate'], 'cps': r['cost_per_success'],
                   'mean_borrows': r['mean_borrows'], 'borrowed_tokens': borrowed_tokens,
                   'credit_efficiency': successes / max(borrowed_tokens, 1) * 1000,
                   'credit_scores': r.get('credit_scores', {})}
            out.append(row)
            print(f"  SR={row['success_rate']:.1%} CPS=${row['cps']:.4f} borrows={row['mean_borrows']:.2f} "
                  f"credit_eff={row['credit_efficiency']:.2f} succ/1K-borrowed scores={row['credit_scores']}")

with open('credit_experiment.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nExported: credit_experiment.json')
