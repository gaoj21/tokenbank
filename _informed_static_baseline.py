"""TB-Informed-Static: Token Bank with consumption priors but no runtime governance.

An internal ablation (NOT an external method): a calibration pass (N
unconstrained tasks) measures per-role consumption, turned into a proportional
static quota plan (quota_r ∝ mean_consumption_r), injected via BudgetAdapter
with gamma=1.0 and ALL runtime governance disabled. It decomposes v2's gains
into "knowing the consumption distribution" (planning-layer information) vs
"adapting during execution" (runtime governance).

  BENCHMARK=FRAMES BUDGETS=8000,16000,32000 .venv/bin/python -u _informed_static_baseline.py
"""
import os
import json
import numpy as np

# ---- load definitions from the experiment script (cells 2 through 10) ----
import re as _re
src = open('_run_v2_experiment.py').read()
_parts = _re.split(r'# ============ NOTEBOOK CELL (\d+) ============\n', src)
cell_map = {int(_parts[i]): _parts[i + 1] for i in range(1, len(_parts), 2)}

for n in [2, 4, 6, 7, 8, 9, 10]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())

# ---- dataset loaders shared with _budget_sweep_vllm.py ----
def load_frames_dataset(n_tasks: int = 824) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset('google/frames-benchmark', split='test')
    tasks = []
    for item in ds:
        q, a = item.get('Prompt'), item.get('Answer')
        if q and a:
            tasks.append({'id': f"frames_{item.get('Unnamed: 0', len(tasks))}",
                          'question': q, 'answer': str(a), 'level': 2,
                          'has_attachment': False,
                          'expected_agents': ['Planner', 'Executor', 'Reviewer']})
    random.shuffle(tasks)
    return tasks[:n_tasks]


def load_musique_dataset(n_tasks: int = 824) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset('dgslibisey/MuSiQue', split='validation')
    tasks = []
    for item in ds:
        if not item.get('answerable', True):
            continue
        q, a = item.get('question'), item.get('answer')
        if q and a:
            tasks.append({'id': f"musique_{item.get('id', len(tasks))}",
                          'question': q, 'answer': str(a), 'level': 2,
                          'has_attachment': False,
                          'expected_agents': ['Planner', 'Executor', 'Reviewer']})
    random.shuffle(tasks)
    return tasks[:n_tasks]


def load_math_dataset(n_tasks: int = 824, levels=('Level 3', 'Level 4', 'Level 5')) -> List[Dict]:
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
                tasks.append({'id': f"math_{subj}_{len(tasks)}",
                              'question': item['problem'], 'answer': ans, 'level': int(item['level'][-1]),
                              'has_attachment': False,
                              'expected_agents': ['Planner', 'Executor', 'Reviewer']})
    random.shuffle(tasks)
    return tasks[:n_tasks]


BENCH = os.getenv('BENCHMARK', 'FRAMES')
BUDGETS_Z = [int(b) for b in os.getenv('BUDGETS', '8000,16000,32000').split(',')]
CALIB_N = int(os.getenv('CALIB_N', '20'))
MODEL_TAG = os.getenv('MODEL_TAG', PRIMARY_MODEL.split('/')[-1])

if BENCH == 'FRAMES':
    TASKS, RUNNER = load_frames_dataset(int(os.getenv('FRAMES_N', '824'))), run_gaia_task_v2
elif BENCH == 'MUSIQUE':
    TASKS, RUNNER = load_musique_dataset(int(os.getenv('MUSIQUE_N', '824'))), run_gaia_task_v2
elif BENCH == 'MATH':
    TASKS, RUNNER = load_math_dataset(int(os.getenv('MATH_N', '824'))), run_gaia_task_v2
elif BENCH == 'GAIA':
    TASKS, RUNNER = load_gaia_dataset(n_tasks=int(os.getenv('GAIA_N', '100'))), run_gaia_task_v2
elif BENCH == 'SWE':
    TASKS, RUNNER = load_swebench_lite(n_tasks=int(os.getenv('SWE_N', '30'))), run_swe_task_v2
else:
    TASKS, RUNNER = FINANCIAL_TASKS, run_financial_task_v2

SEEDS = [int(s) for s in os.getenv('SEEDS', '42').split(',')]
print(f'TB-Informed-Static baseline on {BENCH} ({MODEL_TAG}): {len(TASKS)} tasks, budgets={BUDGETS_Z}, calib_n={CALIB_N}')


def calibrate_role_weights(tasks_calib):
    """Zero-shot-ish calibration: run N tasks unconstrained, measure per-agent
    consumption -> proportional quota weights (water-filling approximation)."""
    cfg = {'name': 'calib', 'budget': 200000, 'policy': 'hard', 'external_plan': None,
           'gamma': 0.0, 'enable_predictive': False, 'enable_borrowing': False,
           'enable_opl': False, 'enable_kv': False, 'theta': 1.0}
    res = run_experiment_v2(tasks_calib, RUNNER, cfg, llm)
    used = {}
    for tr in res['results']:
        for aid, acc in tr['bank_summary']['agents'].items():
            used.setdefault(aid, []).append(acc['used'])
    w = {aid: float(np.mean(v)) for aid, v in used.items()}
    tot = sum(w.values()) or 1.0
    return {aid: w[aid] / tot for aid in w}


results = []
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    weights = calibrate_role_weights(TASKS[:CALIB_N])
    print(f'  calibration weights: { {k: round(v, 3) for k, v in weights.items()} }')
    for B in BUDGETS_Z:
        plan = {aid: max(B // 5, int(B * w)) for aid, w in weights.items()}
        # min-quota bump can push the plan over budget; take the excess back
        # from the largest quota so sum(plan) == B exactly
        excess = sum(plan.values()) - B
        if excess > 0:
            top = max(plan, key=plan.get)
            plan[top] -= excess
        cfg = {'name': 'TB-Informed-Static', 'budget': B, 'min_quota': B // 5,
               'policy': 'hard', 'external_plan': plan, 'gamma': 1.0,
               'enable_predictive': False, 'enable_borrowing': False,
               'enable_opl': False, 'enable_kv': False, 'theta': 1.0}
        print(f"\n--- TB-Informed-Static @ budget={B} seed={seed} plan={plan} ---")
        r = run_experiment_v2(TASKS, RUNNER, cfg, llm)
        results.append({'seed': seed, 'budget': B, 'name': 'TB-Informed-Static', 'plan': plan,
                        'success_rate': r['success_rate'], 'cps': r['cost_per_success'],
                        'mean_tokens': r['mean_tokens'], 'cost': r['total_cost_usd']})
        print(f"  SR={r['success_rate']:.1%} CPS=${r['cost_per_success']:.4f} tokens={r['mean_tokens']:.0f}")

rows = []
for B in BUDGETS_Z:
    sel = [r for r in results if r['budget'] == B]
    rows.append({'budget': B, 'name': 'TB-Informed-Static',
                 'sr_mean': float(np.mean([s['success_rate'] for s in sel])),
                 'sr_std': float(np.std([s['success_rate'] for s in sel])),
                 'cps_mean': float(np.mean([s['cps'] for s in sel])),
                 'tokens_mean': float(np.mean([s['mean_tokens'] for s in sel]))})
    r = rows[-1]
    print(f"budget={B:>6} TB-Informed-Static SR={r['sr_mean']:.1%}±{r['sr_std']:.1%} "
          f"CPS=${r['cps_mean']:.4f} tokens={r['tokens_mean']:.0f}")

out = f'informed_static_{BENCH.lower()}_{MODEL_TAG}.json'
with open(out, 'w') as f:
    json.dump({'model': PRIMARY_MODEL, 'benchmark': BENCH, 'seeds': SEEDS,
               'raw': results, 'agg': rows}, f, indent=2)
print(f'\nExported: {out}')
