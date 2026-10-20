"""Budget sweep (vLLM backend): find the budget regime where TB-v2 dominates.

Reuses dataset + classes + runners from _run_v2_experiment.py (cells 2-10),
then runs {Unconstrained, Static Quota, TB-v2-Full, TB-v2-Scored} x BUDGETS on GAIA.
Unconstrained is budget-independent, so it runs once and is reused.

  SEEDS=42 .venv/bin/python -u _paced_vllm.py
  BUDGETS=8000,16000 SEEDS=42,43,44 .venv/bin/python -u _paced_vllm.py
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

BUDGETS_SWEEP = [int(b) for b in os.getenv('BUDGETS', '2000,4000,8000,16000,32000').split(',')]
SEEDS = [int(s) for s in os.getenv('SEEDS', '42').split(',')]
MODEL_TAG = os.getenv('MODEL_TAG', PRIMARY_MODEL.split('/')[-1])

CONFIGS = [
    # TB-v2-Paced: Full governance + soft-segmented budget pacing. Front 50% of
    # the budget is upfront; the back 50% is released in 3 segments only when
    # >=80% consumed AND a Reviewer checkpoint has passed (fail-fast on
    # runaway tasks; targets the loose-zone CPS bloat where Full >= Unc).
    {'name': 'TB-v2-Paced', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True, 'theta': 0.8,
     'credit_rate': 1.5, 'enable_pacing': True},
    # TB-v2-Paced-v2: consumption-oriented pacing. Tighter upfront (30%),
    # finer segments (k=5, ~1 revision round each), plus the runner progress
    # gate (stall detection; round cap stays at 5 — a hard cap costs SR
    # because some tasks genuinely converge in rounds 4-5). Goal: cut mean
    # token consumption on non-converging tasks without hurting success rate.
    {'name': 'TB-v2-Paced-v2', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True, 'theta': 0.8,
     'credit_rate': 1.5, 'enable_pacing': True, 'pacing_upfront': 0.3, 'pacing_k': 5},
]

UNCONSTRAINED = {'name': 'Unconstrained', 'budget': 200000, 'policy': 'hard', 'external_plan': None,
                 'gamma': 0.0, 'enable_predictive': False, 'enable_borrowing': False, 'enable_opl': False, 'enable_kv': False, 'theta': 1.0}

def load_frames_dataset(n_tasks: int = 824) -> List[Dict]:
    """Google FRAMES: multi-step factoid QA (GAIA-style, public, short answers)."""
    from datasets import load_dataset
    ds = load_dataset('google/frames-benchmark', split='test')
    tasks = []
    for item in ds:
        q, a = item.get('Prompt'), item.get('Answer')
        if q and a:
            tasks.append({
                'id': f"frames_{item.get('Unnamed: 0', len(tasks))}",
                'question': q, 'answer': str(a), 'level': 2,
                'has_attachment': False,
                'expected_agents': ['Planner', 'Executor', 'Reviewer']
            })
    random.shuffle(tasks)
    return tasks[:n_tasks]


def load_musique_dataset(n_tasks: int = 824) -> List[Dict]:
    """MuSiQue: 2-4 hop compositional QA (GAIA-style, public, short answers)."""
    from datasets import load_dataset
    ds = load_dataset('dgslibisey/MuSiQue', split='validation')
    tasks = []
    for item in ds:
        if not item.get('answerable', True):
            continue
        q, a = item.get('question'), item.get('answer')
        if q and a:
            tasks.append({
                'id': f"musique_{item.get('id', len(tasks))}",
                'question': q, 'answer': str(a), 'level': 2,
                'has_attachment': False,
                'expected_agents': ['Planner', 'Executor', 'Reviewer']
            })
    random.shuffle(tasks)
    return tasks[:n_tasks]


def load_math_dataset(n_tasks: int = 824, levels=('Level 3', 'Level 4', 'Level 5')) -> List[Dict]:
    """MATH (Hendrycks): competition math L3-5, numeric/expression answers from \\boxed{}."""
    from datasets import load_dataset
    subjects = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
                'number_theory', 'prealgebra', 'precalculus']

    def _boxed(sol: str) -> str:
        i = sol.rfind('\\boxed{')
        if i < 0:
            return ''
        j, depth = i + 7, 1
        while j < len(sol) and depth:
            depth += {'{': 1, '}': -1}.get(sol[j], 0)
            j += 1
        return sol[i + 7:j - 1]

    tasks = []
    for subj in subjects:
        try:
            ds = load_dataset('EleutherAI/hendrycks_math', subj, split='test')
        except Exception as e:
            print(f'MATH {subj} load failed: {e}')
            continue
        for item in ds:
            if item.get('level') not in levels:
                continue
            ans = _boxed(item.get('solution', ''))
            if item.get('problem') and ans:
                tasks.append({
                    'id': f"math_{subj}_{len(tasks)}",
                    'question': item['problem'], 'answer': ans, 'level': int(item['level'][-1]),
                    'has_attachment': False,
                    'expected_agents': ['Planner', 'Executor', 'Reviewer']
                })
    random.shuffle(tasks)
    print(f'MATH L3-5 pool: {len(tasks)} tasks (7 subjects)')
    return tasks[:n_tasks]


BENCH = os.getenv('BENCHMARK', 'GAIA')
if BENCH == 'SWE':
    # SWE_N controls task count (default 30; 300 = full SWE-bench Lite test split)
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
    # GAIA_N reload: real GAIA (gated, needs HF_TOKEN) up to full L1-2 non-media set
    TASKS = load_gaia_dataset(n_tasks=int(os.getenv('GAIA_N', '100')))
    RUNNER = run_gaia_task_v2
print(f'Budget sweep on {BENCH} ({MODEL_TAG}): {len(TASKS)} tasks x {len(CONFIGS)} configs '
      f'x {len(BUDGETS_SWEEP)} budgets x {len(SEEDS)} seeds + Unconstrained x{len(SEEDS)}')

sweep = []
SKIP_UNCONSTRAINED = os.getenv('SKIP_UNCONSTRAINED', '0') == '1'
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    # Unconstrained reference: budget-independent, run once per seed
    if SKIP_UNCONSTRAINED:
        print('\n--- Unconstrained skipped (SKIP_UNCONSTRAINED=1, reuse prior reference) ---')
    else:
        print(f"\n--- Unconstrained @ budget=200000 seed={seed} ---")
        r = run_experiment_v2(TASKS, RUNNER, dict(UNCONSTRAINED), llm)
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
            sweep.append({'seed': seed, 'budget': budget, 'name': c['name'],
                          'success_rate': r['success_rate'], 'cps': r['cost_per_success'],
                          'mean_tokens': r['mean_tokens'], 'cost': r['total_cost_usd'],
                          'mean_borrows': r['mean_borrows'], 'recovery_rate': r['recovery_rate']})
            print(f"  SR={r['success_rate']:.1%} CPS=${r['cost_per_success']:.4f} tokens={r['mean_tokens']:.0f} "
                  f"borrows={r['mean_borrows']:.2f} recovery={r['recovery_rate']:.1%}")

# aggregate over seeds
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
        print(f"budget={budget:>6} {name:>14} SR={r['sr_mean']:.1%}±{r['sr_std']:.1%} CPS=${r['cps_mean']:.4f} tokens={r['tokens_mean']:.0f}")

_prefix = 'paced' if BENCH == 'GAIA' else f'paced_{BENCH.lower()}'
_tag = os.getenv('OUT_TAG', '')
out_json = f'{_prefix}_{MODEL_TAG}{_tag}.json'
out_png = f'{_prefix}_{MODEL_TAG}{_tag}.png'
with open(out_json, 'w') as f:
    json.dump({'model': PRIMARY_MODEL, 'seeds': SEEDS, 'raw': sweep, 'agg': rows}, f, indent=2)

# plot SR and CPS vs budget
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = {'Unconstrained': '#ef4444', 'TB-v2-Paced': '#a855f7', 'TB-v2-Paced-v2': '#0ea5e9'}
for name, color in colors.items():
    sel = [r for r in rows if r['name'] == name]
    x = [r['budget'] for r in sel]
    ax1.errorbar(x, [r['sr_mean'] * 100 for r in sel], yerr=[r['sr_std'] * 100 for r in sel],
                 marker='o', linewidth=2.5, capsize=5, label=name, color=color)
    ax2.plot(x, [r['cps_mean'] for r in sel], marker='s', linewidth=2.5, label=name, color=color)
for ax, title, ylabel in [(ax1, f'Success Rate vs Budget (GAIA, {MODEL_TAG})', 'Success Rate (%)'),
                          (ax2, f'Cost per Success vs Budget (GAIA, {MODEL_TAG})', 'CPS (USD)')]:
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
