"""Adaptive runaway guard ablation (DeepSeek flash backend).

Tests the new strategy: pacing + stall only activate for tasks that are both
stuck (answer unchanged across rounds) AND have burned a significant fraction
of the budget. Normal converging tasks keep the full budget.

Reuses dataset + classes + runners from _run_v2_experiment.py (cells 2-10).

  LLM_BACKEND=deepseek DEEPSEEK_MODEL=deepseek-v4-flash \
  BUDGETS=32000 SEEDS=42,43 OUT_TAG=_adaptive32k \
  HF_TOKEN=hf_xxx .venv/bin/python -u _adaptive_ablation.py
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

# Load cells in order, but seed before cell 4 (dataset loading) so the
# sampled tasks are deterministic across repeated smoke runs.
for n in [2]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())
random.seed(int(os.getenv('DATA_SEED', '42')))
np.random.seed(int(os.getenv('DATA_SEED', '42')))
for n in [4, 6, 7, 8, 9, 10]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())

BUDGETS_SWEEP = [int(b) for b in os.getenv('BUDGETS', '32000').split(',')]
SEEDS = [int(s) for s in os.getenv('SEEDS', '42').split(',')]
MODEL_TAG = os.getenv('MODEL_TAG', PRIMARY_MODEL.split('/')[-1])

CONFIGS = [
    # Baseline: hard pacing with stall detection (the strategy that lost SR).
    {'name': 'Full+Stall', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True,
     'enable_opl': True, 'enable_kv': True, 'theta': 0.8,
     'credit_rate': 1.5, 'enable_pacing': True,
     'pacing_upfront': 0.95, 'pacing_k': 1, 'pacing_max_rounds': 5},
    # New: start with full budget; only clamp down on runaway tasks.
    {'name': 'TB-v2-Adaptive', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True,
     'enable_opl': True, 'enable_kv': True, 'theta': 0.8,
     'credit_rate': 1.5, 'enable_pacing': True,
     'pacing_upfront': 0.5, 'pacing_k': 3, 'pacing_max_rounds': 5,
     'adaptive_runaway': True, 'runaway_token_fraction': 0.25,
     'runaway_rounds': 1},
]

UNCONSTRAINED = {'name': 'Unconstrained', 'budget': 200000, 'policy': 'hard',
                 'external_plan': None, 'gamma': 0.0,
                 'enable_predictive': False, 'enable_borrowing': False,
                 'enable_opl': False, 'enable_kv': False, 'theta': 1.0}

BENCH = os.getenv('BENCHMARK', 'GAIA')
if BENCH == 'SWE':
    TASKS = load_swebench_lite(n_tasks=int(os.getenv('SWE_N', '30')))
    RUNNER = run_swe_task_v2
elif BENCH == 'FRAMES':
    TASKS = load_frames_dataset(n_tasks=int(os.getenv('FRAMES_N', '824')))
    RUNNER = run_gaia_task_v2
elif BENCH == 'MUSIQUE':
    TASKS = load_musique_dataset(n_tasks=int(os.getenv('MUSIQUE_N', '824')))
    RUNNER = run_gaia_task_v2
elif BENCH == 'MATH':
    TASKS = load_math_dataset(n_tasks=int(os.getenv('MATH_N', '824')))
    RUNNER = run_gaia_task_v2
elif BENCH == 'FINANCIAL':
    TASKS, RUNNER = FINANCIAL_TASKS, run_financial_task_v2
else:
    TASKS = load_gaia_dataset(n_tasks=int(os.getenv('GAIA_N', '100')))
    RUNNER = run_gaia_task_v2

print(f'Adaptive ablation on {BENCH} ({MODEL_TAG}): {len(TASKS)} tasks x '
      f'{len(CONFIGS)} configs x {len(BUDGETS_SWEEP)} budgets x {len(SEEDS)} seeds '
      f'+ Unconstrained x{len(SEEDS)}')


def dump_per_task(r, seed, budget, name, fh):
    """One JSONL row per task for paired analysis."""
    for tr in r['results']:
        bs = tr['bank_summary']
        calls = tr.get('agent_calls', [])
        fh.write(json.dumps({
            'seed': seed, 'budget': budget, 'config': name, 'task_id': tr['task_id'],
            'success': tr['success'], 'total_used': bs['total_used'],
            'cost_usd': bs['total_cost_usd'], 'borrows': bs['borrow_count'],
            'summarizations': bs.get('summarization_count', 0),
            'pacing_releases': bs.get('pacing_releases', 0),
            'runaway_active': bs.get('runaway_active', False),
            'n_calls': len(calls), 'n_blocked': sum(1 for c in calls if not c['success']),
            'stalled': tr.get('stalled', False),
        }) + '\n')
    fh.flush()


_pt_fh = open(f'tasks_{MODEL_TAG}{os.getenv("OUT_TAG", "")}.jsonl', 'a')
sweep = []
SKIP_UNCONSTRAINED = os.getenv('SKIP_UNCONSTRAINED', '0') == '1'
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    if SKIP_UNCONSTRAINED:
        print('\n--- Unconstrained skipped (SKIP_UNCONSTRAINED=1, reuse prior reference) ---')
    else:
        print(f"\n--- Unconstrained @ budget=200000 seed={seed} ---")
        r = run_experiment_v2(TASKS, RUNNER, dict(UNCONSTRAINED), llm)
        dump_per_task(r, seed, BUDGETS_SWEEP[0], 'Unconstrained', _pt_fh)
        for budget in BUDGETS_SWEEP:
            sweep.append({'seed': seed, 'budget': budget, 'name': 'Unconstrained',
                          'success_rate': r['success_rate'], 'cps': r['cost_per_success'],
                          'mean_tokens': r['mean_tokens'], 'cost': r['total_cost_usd']})
        print(f"  SR={r['success_rate']:.1%} CPS=${r['cost_per_success']:.4f} tokens={r['mean_tokens']:.0f}")
    for budget in BUDGETS_SWEEP:
        for cfg in CONFIGS:
            c = dict(cfg)
            c['budget'] = budget
            c['min_quota'] = budget // 5
            print(f"\n--- {c['name']} @ budget={budget} seed={seed} ---")
            r = run_experiment_v2(TASKS, RUNNER, c, llm)
            dump_per_task(r, seed, budget, c['name'], _pt_fh)
            sweep.append({'seed': seed, 'budget': budget, 'name': c['name'],
                          'success_rate': r['success_rate'], 'cps': r['cost_per_success'],
                          'mean_tokens': r['mean_tokens'], 'cost': r['total_cost_usd'],
                          'mean_borrows': r['mean_borrows'],
                          'recovery_rate': r['recovery_rate']})
            print(f"  SR={r['success_rate']:.1%} CPS=${r['cost_per_success']:.4f} "
                  f"tokens={r['mean_tokens']:.0f} borrows={r['mean_borrows']:.2f} "
                  f"recovery={r['recovery_rate']:.1%}")

names = ([] if SKIP_UNCONSTRAINED else ['Unconstrained']) + [c['name'] for c in CONFIGS]
rows = []
for budget in BUDGETS_SWEEP:
    for name in names:
        sel = [s for s in sweep if s['budget'] == budget and s['name'] == name]
        if not sel:
            continue
        rows.append({'budget': budget, 'name': name,
                     'sr_mean': float(np.mean([s['success_rate'] for s in sel])),
                     'sr_std': float(np.std([s['success_rate'] for s in sel])),
                     'cps_mean': float(np.mean([s['cps'] for s in sel])),
                     'tokens_mean': float(np.mean([s['mean_tokens'] for s in sel]))})
        r = rows[-1]
        print(f"budget={budget:>6} {name:>14} SR={r['sr_mean']:.1%}±{r['sr_std']:.1%} "
              f"CPS=${r['cps_mean']:.4f} tokens={r['tokens_mean']:.0f}")

_prefix = 'adaptive' if BENCH == 'GAIA' else f'adaptive_{BENCH.lower()}'
_tag = os.getenv('OUT_TAG', '')
out_json = f'{_prefix}_{MODEL_TAG}{_tag}.json'
out_png = f'{_prefix}_{MODEL_TAG}{_tag}.png'
with open(out_json, 'w') as f:
    json.dump({'model': PRIMARY_MODEL, 'seeds': SEEDS, 'raw': sweep, 'agg': rows}, f, indent=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = {'Unconstrained': '#ef4444', 'Full+Stall': '#a855f7',
          'TB-v2-Adaptive': '#0ea5e9'}
for name, color in colors.items():
    sel = [r for r in rows if r['name'] == name]
    x = [r['budget'] for r in sel]
    ax1.errorbar(x, [r['sr_mean'] * 100 for r in sel], yerr=[r['sr_std'] * 100 for r in sel],
                 marker='o', linewidth=2.5, capsize=5, label=name, color=color)
    ax2.plot(x, [r['cps_mean'] for r in sel], marker='s', linewidth=2.5, label=name, color=color)
for ax, title, ylabel in [(ax1, f'Success Rate vs Budget ({BENCH}, {MODEL_TAG})', 'Success Rate (%)'),
                          (ax2, f'Cost per Success vs Budget ({BENCH}, {MODEL_TAG})', 'CPS (USD)')]:
    ax.set_xscale('log')
    ax.set_xlabel('Per-task budget (tokens)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
ax2.set_yscale('log')
plt.tight_layout()
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f'\nExported: {out_json}, {out_png}')
