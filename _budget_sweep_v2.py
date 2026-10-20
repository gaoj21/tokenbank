"""Budget-constrained sweep on full GAIA (seed 42, temp=0).

Story: given a binding token budget, which allocation method achieves the
best success rate, and how many tokens does it actually spend?

Methods per budget:
  - Static Quota : naive equal split, hard policy (baseline allocation)
  - TB-v2-Full   : quotas + predictive rebalancing + OPL + KV + borrowing
                   (no pacing/runaway/stall)
  - Full+Stall   : TB-v2-Full + adaptive runaway guard + stall (frac=0.2)

  LLM_BACKEND=deepseek DEEPSEEK_MODEL=deepseek-v4-flash HF_TOKEN=hf_xxx \
  BUDGETS=8000 .venv/bin/python -u _budget_sweep_v2.py
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

# Seed before dataset loading so sampled tasks are deterministic across runs.
for n in [2]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())
random.seed(int(os.getenv('DATA_SEED', '42')))
np.random.seed(int(os.getenv('DATA_SEED', '42')))
for n in [4, 6, 7, 8, 9, 10]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())

BUDGETS_SWEEP = [int(b) for b in os.getenv('BUDGETS', '4000,8000,16000,32000').split(',')]
SEEDS = [int(s) for s in os.getenv('SEEDS', '42').split(',')]
MODEL_TAG = os.getenv('MODEL_TAG', PRIMARY_MODEL.split('/')[-1])

def load_math_dataset(n_tasks: int = 500, level_filter=None):
    """MATH-500 (OpenAI's standard MATH test subset). Fields: problem, solution,
    answer, subject, level. Full MATH test (5000) is out of budget scope."""
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    tasks = []
    for i, item in enumerate(ds):
        lvl = item.get('level', '')
        if level_filter and lvl not in level_filter:
            continue
        tasks.append({'id': f'math_{i}', 'question': item['problem'],
                      'answer': str(item['answer']), 'level': lvl,
                      'expected_agents': ['Planner', 'Executor', 'Reviewer']})
    random.shuffle(tasks)
    return tasks[:n_tasks]


def load_bbeh_dataset(per_task: int = 38):
    """BIG-Bench Extra Hard (23 reasoning tasks). A fixed per-task sample in a fixed order (first
    `per_task` items of each task after a seeded shuffle), so the dev/test split below is the same
    for every model; 23 x 38 = 874 items, of which the test split keeps 80%."""
    from datasets import load_dataset
    ds = load_dataset('BBEH/bbeh', split='train')
    by_task = {}
    for i, item in enumerate(ds):
        by_task.setdefault(item['task'], []).append((i, item))
    tasks = []
    for name in sorted(by_task):
        items = by_task[name][:]
        random.Random(f'bbeh-{name}').shuffle(items)
        for i, item in sorted(items[:per_task], key=lambda z: z[0]):
            tasks.append({'id': f"bbeh_{name.replace(' ', '_')}_{i}", 'question': item['input'],
                          'answer': str(item['target']), 'subtask': name,
                          'expected_agents': ['Planner', 'Executor', 'Reviewer']})
    return tasks


def load_musique_dataset(n_tasks: int = 500):
    """MuSiQue validation set (2417 multi-hop QA). Fields: id, question,
    answer. Only answerable items are kept."""
    from datasets import load_dataset
    ds = load_dataset('dgslibisey/MuSiQue', split='validation')
    tasks = []
    for item in ds:
        if not item.get('answerable', True):
            continue
        tasks.append({'id': f"musique_{item['id']}", 'question': item['question'],
                      'answer': str(item['answer']),
                      'expected_agents': ['Planner', 'Executor', 'Reviewer']})
    random.shuffle(tasks)
    return tasks[:n_tasks]



def load_swe_v5(n_tasks: int = 300):
    """SWE-bench Lite for the long-horizon compaction comparison.

    The task is presented as: problem statement + the buggy code region, with
    the gold patch's added lines as the target. Two honest limitations:
      * localization is GIVEN. The buggy region is reconstructed from the gold
        patch's context and removed lines, so the agent is not asked to find the
        file. A full SWE-bench agent must locate it itself.
      * judging is a token-F1 PROXY, not the official FAIL_TO_PASS /
        PASS_TO_PASS test execution. Never compare these numbers to published
        SWE-bench resolve rates.
    Both are lifted in the execution-based variant.
    """
    from datasets import load_dataset
    ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    tasks = []
    for item in ds:
        patch = item.get('patch', '') or ''
        added = '\n'.join(l[1:] for l in patch.split('\n')
                          if l.startswith('+') and not l.startswith('+++'))
        if not added.strip():
            continue
        ctx = _swe_context_from_gold_patch(patch)
        tasks.append({
            'id': f"swe_{item['instance_id']}",
            'question': (f"Repository: {item.get('repo','')}\n"
                         f"Issue:\n{item.get('problem_statement','')[:3000]}\n\n"
                         f"Current code:\n```python\n{ctx[:4000]}\n```\n"
                         f"Produce the corrected code."),
            'answer': added,
            'expected_agents': ['Planner', 'Executor', 'Reviewer'],
        })
    random.shuffle(tasks)
    return tasks[:n_tasks]


def load_swe_oracle(n_tasks: int = 500):
    """SWE-bench Verified, oracle-localization, producing EXECUTABLE patches.

    Differs from load_swe_v5 in the two ways that made the earlier runs
    unusable for official evaluation:
      * the reply is a set of labelled code regions, which _swe_oracle turns
        into a unified diff mechanically, so `git apply` accepts it;
      * scoring is the official FAIL_TO_PASS / PASS_TO_PASS test execution,
        run separately from this sweep, not a token-F1 proxy.
    Localization is still given (the regions to fix are shown), which is a real
    limitation and must be stated: these are oracle-localization numbers and are
    not comparable to agentic SWE-bench resolve rates.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _swe_oracle import render_task
    from datasets import load_dataset
    ds = load_dataset('SWE-bench/SWE-bench_Verified', split='test')
    tasks = [{'id': it['instance_id'], 'question': render_task(it),
              'answer': '',            # judged externally by the harness
              'expected_agents': ['Planner', 'Executor', 'Reviewer']}
             for it in ds]
    random.shuffle(tasks)
    return tasks[:n_tasks]


def load_bigcodebench(n_tasks: int = 500):
    """BigCodeBench: library-heavy function synthesis, judged by execution.

    Added because the other benchmarks leave one carried slot untested. Segment
    accounting puts raw tool output at 0.3-1.0% of characters on MATH and GAIA,
    so the mechanism that bounds it has never faced a workload where it matters.
    Here the average task imports two or more of 139 libraries, so the executor
    runs code, reads tracebacks and retries, which is exactly the traffic that
    slot carries. It also judges by running the task's own unittest suite with no
    oracle assistance, which the SWE-bench setup, with localization given, cannot.
    """
    from datasets import load_dataset
    ds = load_dataset('bigcode/bigcodebench', split='v0.1.4')
    tasks = [{'id': f"bcb_{it['task_id']}",
              'question': (it['instruct_prompt'] +
                           '\n\nWrite the complete function in a single ```python block. '
                           'Keep the exact signature and name `task_func`.'),
              'answer': it['test'],          # the unittest suite is the ground truth
              'expected_agents': ['Planner', 'Executor', 'Reviewer']}
             for it in ds]
    tasks.sort(key=lambda t: t['id'])          # fixed order: the dev/test split is identical for every model
    return tasks[:n_tasks]


def load_livecodebench(n_tasks: int = 100000):
    """LiveCodeBench code_generation_lite release_v6, prepared by analysis/lcb_prepare.py. The answer
    field carries the test spec; the tests themselves are read from data/lcb/tests/ when judged."""
    tasks = []
    for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'lcb', 'tasks.jsonl')):
        it = json.loads(l)
        if it['starter']:
            q = (f"{it['content']}\n\nComplete the following starter code, keeping the class and method signature:\n"
                 f"```python\n{it['starter']}\n```\nReturn the complete solution in one ```python block.")
        else:
            q = (f"{it['content']}\n\nWrite a Python 3 program that reads the input from standard input and writes "
                 f"the answer to standard output. Return the complete program in one ```python block.")
        tasks.append({'id': it['id'], 'question': q, 'expected_agents': ['Planner', 'Executor', 'Reviewer'],
                      'answer': json.dumps({'id': it['id'], 'testtype': it['testtype'], 'func_name': it['func_name']})})
    tasks.sort(key=lambda t: t['id'])          # fixed order: the dev/test split is identical for every model
    return tasks[:n_tasks]


def load_mbppplus():
    """MBPP+ (EvalPlus, 378 problems): the description plus its first example assertion, which names the
    function. Judged by the task's MBPP+ test script (_mbpp_judge)."""
    import pandas as pd
    d = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mbppplus', 'test.parquet'))
    tasks = []
    for r in d.itertuples():
        ex = list(r.test_list)[0]
        q = (f"{r.prompt}\nYour function must satisfy this example:\n{ex}\n"
             f"Return the complete function (with any imports it needs) in one ```python block.")
        tasks.append({'id': f"mbpp_{r.task_id}", 'question': q, 'answer': f"mbpp_{r.task_id}",
                      'expected_agents': ['Planner', 'Executor', 'Reviewer']})
    tasks.sort(key=lambda t: t['id'])
    return tasks


def _bbh_format(task, target):
    """The answer format, stated as BBH's few-shot exemplars show it, derived from the target's type; without
    it models answer in their own words ("implausible", a date instead of an option letter)."""
    import re as _re
    if task == 'word_sorting':
        return 'Answer with the sorted words separated by single spaces, e.g. ANSWER: apple banana cherry.'
    if task == 'dyck_languages':
        return 'Answer with only the closing brackets that complete the sequence, separated by spaces, e.g. ANSWER: ] ) >.'
    if _re.fullmatch(r'\([A-Z]\)', target):
        return 'Answer with the option letter, e.g. ANSWER: (A).'
    t = target.lower()
    if t in ('yes', 'no'):
        return 'Answer Yes or No, e.g. ANSWER: Yes.'
    if t in ('valid', 'invalid'):
        return 'Answer valid or invalid, e.g. ANSWER: valid.'
    if t in ('true', 'false'):
        return 'Answer True or False, e.g. ANSWER: True.'
    if _re.fullmatch(r'-?\d+', target):
        return 'Answer with the number only, e.g. ANSWER: 7.'
    return 'Answer with the final answer only.'


def load_bbh(per_task: int = 37):
    """BIG-Bench Hard: a fixed sample of per_task problems from each of its 27 tasks (seeded per task,
    so every model sees the same items), scored by the official-style fuzzy match (_bbeh_judge)."""
    import glob, pandas as pd
    tasks = []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'bbh', '*', 'test-*.parquet'))):
        name = os.path.basename(os.path.dirname(f)); d = pd.read_parquet(f)
        idx = sorted(random.Random(f'bbh-{name}').sample(range(len(d)), min(per_task, len(d))))
        for i in idx:
            tasks.append({'id': f'bbh_{name}_{i}', 'question': d.input.iloc[i] + '\n\n' + _bbh_format(name, str(d.target.iloc[i])),
                          'answer': str(d.target.iloc[i]),
                          'expected_agents': ['Planner', 'Executor', 'Reviewer']})
    tasks.sort(key=lambda t: t['id'])
    return tasks


BENCH = os.getenv('BENCHMARK', 'GAIA')
if BENCH == 'BBH':
    TASKS = load_bbh(int(os.getenv('BBH_PER_TASK', '37')))
elif BENCH == 'MBPP':
    TASKS = load_mbppplus()
elif BENCH == 'LCB':
    TASKS = load_livecodebench()
elif BENCH == 'BCB':
    TASKS = load_bigcodebench(n_tasks=int(os.getenv('BCB_N', '100000')))
elif BENCH == 'SWEO':
    TASKS = load_swe_oracle(n_tasks=int(os.getenv('SWE_N', '500')))
elif BENCH == 'SWE':
    TASKS = load_swe_v5(n_tasks=int(os.getenv('SWE_N', '300')))
elif BENCH == 'FRAMES':
    TASKS = FRAMES_TASKS[:int(os.getenv('FRAMES_N', str(len(FRAMES_TASKS))))]
elif BENCH == 'BBEH':
    TASKS = load_bbeh_dataset(per_task=int(os.getenv('BBEH_PER_TASK', '38')))
elif BENCH == 'MUSIQUE':
    TASKS = load_musique_dataset(n_tasks=int(os.getenv('MUSIQUE_N', '500')))
elif BENCH == 'MATH':
    _mlvls = os.getenv('MATH_LEVELS')
    TASKS = load_math_dataset(n_tasks=int(os.getenv('MATH_N', '500')),
                              level_filter=_mlvls.split(',') if _mlvls else None)
else:
    _lvls = [int(x) for x in os.getenv('LEVEL_FILTER', '1,2,3').split(',')]
    TASKS = load_gaia_dataset(n_tasks=int(os.getenv('GAIA_N', '1000')), level_filter=_lvls)
# v5 P1.2: fixed dev/test split. All tuning happens on dev (20%); test (80%)
# is touched only for final tables. Split uses its own seeded shuffle so it is
# independent of the loaders' internal sampling order.
SPLIT = os.getenv('SPLIT', 'full')
if SPLIT != 'full':
    _tt = list(TASKS)
    random.Random(42).shuffle(_tt)
    _k = max(1, int(0.2 * len(_tt)))
    TASKS = _tt[:_k] if SPLIT == 'dev' else _tt[_k:]
    print(f'SPLIT={SPLIT}: using {len(TASKS)}/{len(_tt)} tasks')
# Restrict to an explicit list of task ids (the test half of the live CLIMB evaluation, whose
# thresholds were certified on the other half's offline records).
if os.getenv('TASK_IDS_FILE'):
    _ids = set(json.load(open(os.getenv('TASK_IDS_FILE'))))
    TASKS = [t for t in TASKS if t['id'] in _ids]
    print(f'TASK_IDS_FILE: using {len(TASKS)} tasks')
# Open book: MuSiQue with its own paragraphs, FRAMES with the Wikipedia articles it links (_open_book.py).
if os.getenv('OPEN_BOOK') == '1' and BENCH in ('FRAMES', 'MUSIQUE'):
    import _open_book
    TASKS = [dict(t) for t in TASKS]
    TASKS = _open_book.frames(TASKS) if BENCH == 'FRAMES' else _open_book.musique(TASKS)
RUNNER = run_gaia_task_v2

CONFIGS = [
    {'name': 'Unconstrained', 'budget': 200000, 'policy': 'hard', 'external_plan': None,
     'gamma': 0.0, 'enable_predictive': False, 'enable_borrowing': False,
     'enable_opl': False, 'enable_kv': False, 'theta': 1.0},
    # v5 P1.3: the TRUE upper bound — no governance but with the context-hygiene
    # fixes (⑥⑦). The old Unconstrained uses the legacy ever-growing prompt, so
    # "IC beats Unconstrained" confounded governance with context hygiene.
    {'name': 'Unc+Compact', 'budget': 200000, 'policy': 'hard', 'external_plan': None,
     'gamma': 0.0, 'enable_predictive': False, 'enable_borrowing': False,
     'enable_opl': False, 'enable_kv': True, 'theta': 1.0,
     'kv_trigger': 0.65, 'exec_compact': True},
    {'name': 'Static Quota', 'policy': 'hard', 'external_plan': None,
     'gamma': 1.0, 'enable_predictive': False, 'enable_borrowing': False,
     'enable_opl': False, 'enable_kv': False, 'theta': 1.0},
    {'name': 'TB-v2-Full', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True,
     'enable_opl': True, 'enable_kv': True, 'theta': 0.8, 'credit_rate': 1.5},
    {'name': 'Full+Stall', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True,
     'enable_opl': True, 'enable_kv': True, 'theta': 0.8, 'credit_rate': 1.5,
     'enable_pacing': True, 'pacing_upfront': 0.95, 'pacing_k': 1,
     'pacing_max_rounds': 5, 'adaptive_runaway': True,
     'runaway_token_fraction': float(os.getenv('RUNAWAY_FRAC', '0.2')),
     'runaway_rounds': int(os.getenv('RUNAWAY_ROUNDS', '1')),
     'runaway_min_tokens': int(os.getenv('RUNAWAY_MIN_TOKENS', '0'))},
    # v3: Full+Stall + all five improvements (① final reserve, ② role-scaled
    # credit lines, ③ predictive dispatch estimates, ④ similarity stall +
    # budget-adaptive runaway fraction, ⑤ predictive lender selection).
    {'name': 'Improved', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True,
     'enable_opl': True, 'enable_kv': True, 'theta': 0.8, 'credit_rate': 1.5,
     'enable_pacing': True, 'pacing_upfront': 0.95, 'pacing_k': 1,
     'pacing_max_rounds': 5, 'adaptive_runaway': True,
     'runaway_rounds': int(os.getenv('RUNAWAY_ROUNDS', '1')),
     'runaway_min_tokens': int(os.getenv('RUNAWAY_MIN_TOKENS', '0')),
     'final_reserve': 512,
     'role_credit_mult': {'Planner': 0.5, 'Executor': 1.5, 'Reviewer': 1.0, 'Debugger': 0.75},
     'predictive_est': True, 'stall_sim': 0.9,
     'runaway_frac_auto': True, 'predictive_lending': True},
    # v4 (direction 3): attack prompt growth, not post-hoc rescue.
    #   Compact       = Full+Stall + earlier KV trigger (0.65) + executor
    #                   context compaction (latest answer+feedback only,
    #                   tool output capped at 800 chars).
    #   Improved+Compact = Improved + the same two levers.
    {'name': 'Compact', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True,
     'enable_opl': True, 'enable_kv': True, 'theta': 0.8, 'credit_rate': 1.5,
     'enable_pacing': True, 'pacing_upfront': 0.95, 'pacing_k': 1,
     'pacing_max_rounds': 5, 'adaptive_runaway': True,
     'runaway_token_fraction': float(os.getenv('RUNAWAY_FRAC', '0.2')),
     'runaway_rounds': int(os.getenv('RUNAWAY_ROUNDS', '1')),
     'runaway_min_tokens': int(os.getenv('RUNAWAY_MIN_TOKENS', '0')),
     'kv_trigger': 0.65, 'exec_compact': True},
    {'name': 'Improved+Compact', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True,
     'enable_opl': True, 'enable_kv': True, 'theta': 0.8, 'credit_rate': 1.5,
     'enable_pacing': True, 'pacing_upfront': 0.95, 'pacing_k': 1,
     'pacing_max_rounds': 5, 'adaptive_runaway': True,
     'runaway_rounds': int(os.getenv('RUNAWAY_ROUNDS', '1')),
     'runaway_min_tokens': int(os.getenv('RUNAWAY_MIN_TOKENS', '0')),
     'final_reserve': 512,
     'role_credit_mult': {'Planner': 0.5, 'Executor': 1.5, 'Reviewer': 1.0, 'Debugger': 0.75},
     'predictive_est': True, 'stall_sim': 0.9,
     'runaway_frac_auto': True, 'predictive_lending': True,
     'kv_trigger': 0.65, 'exec_compact': True},
    # v4b: 16K-tuned combo. Diagnosis @16K: final_reserve=512 taxes every task
    # 512 tokens (3.2% of budget) but the closing call only fired ~9 times and
    # rarely converted (blocked 87->122, borrows 191->147 vs plain Compact);
    # stall_sim=0.9 over-triggers (17 vs 14 stalls, no extra successes).
    # Fix: halve the reserve tax, make "unchanged" stricter (0.95 Jaccard).
    {'name': 'Combo16k', 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True,
     'enable_opl': True, 'enable_kv': True, 'theta': 0.8, 'credit_rate': 1.5,
     'enable_pacing': True, 'pacing_upfront': 0.95, 'pacing_k': 1,
     'pacing_max_rounds': 5, 'adaptive_runaway': True,
     'runaway_rounds': int(os.getenv('RUNAWAY_ROUNDS', '1')),
     'runaway_min_tokens': int(os.getenv('RUNAWAY_MIN_TOKENS', '0')),
     'final_reserve': 256,
     'role_credit_mult': {'Planner': 0.5, 'Executor': 1.5, 'Reviewer': 1.0, 'Debugger': 0.75},
     'predictive_est': True, 'stall_sim': 0.95,
     'runaway_frac_auto': True, 'predictive_lending': True,
     'kv_trigger': 0.65, 'exec_compact': True},
    # Published-style baselines:
    #   EarlyStop-2: static quotas + hard 2-round cap (naive early stopping)
    #   BudgetForcing: static quotas + forced final answer on quota block
    #     (s1-style budget forcing adapted to the multi-agent harness)
    {'name': 'EarlyStop-2', 'policy': 'hard', 'external_plan': None,
     'gamma': 1.0, 'enable_predictive': False, 'enable_borrowing': False,
     'enable_opl': False, 'enable_kv': False, 'theta': 1.0, 'max_rounds': 2},
    {'name': 'BudgetForcing', 'policy': 'hard', 'external_plan': None,
     'gamma': 1.0, 'enable_predictive': False, 'enable_borrowing': False,
     'enable_opl': False, 'enable_kv': False, 'theta': 1.0,
     'force_final_on_block': True},
]

# ==================== v5 Phase 2 configs (enabled with V5=1) ====================
# M0: dead mechanisms (borrowing / role_credit / predictive / stall) are gone
# by construction — v5 configs never enable them. final_reserve kept.
V5_THRESH = {  # env-overridable thresholds; defaults tuned offline on dev
               # (GAIA-dev: SR=baseline, -75% tok; MuSiQue-dev: +0.8pp, -61% tok)
    'tau_conf': float(os.getenv('TAU_CONF', '-0.05')),
    'tau_high': float(os.getenv('TAU_HIGH', '0.6')),
    'theta0': float(os.getenv('THETA0', '-0.10')),
    'theta1': float(os.getenv('THETA1', '-0.15')),
}
V5_BASE = {'policy': 'hard', 'external_plan': None, 'gamma': 0.0,
           'enable_predictive': False, 'enable_borrowing': False,
           'enable_opl': False, 'enable_kv': False, 'theta': 1.0,
           'capture_logprobs': True, 'final_reserve': 256, 'v5': True,
           'exec_compact': True, **V5_THRESH}
V5_FULL = {'m1_early_stop': True, 'm2_cascade': True,
           'm3_structured_reviewer': True, 'm3_tool_notes': True,
           'm3_plan_freeze': True, 'm3_answer_format': True,
           'm5_strategy': os.getenv('M5_STRATEGY', 'hybrid')}
CONFIGS_V5 = [
    {**V5_BASE, **V5_FULL, 'name': 'V5-A0-Full', 'm4_pool': True},
    {**V5_BASE, **V5_FULL, 'name': 'V5-A1-NoM1', 'm1_early_stop': False, 'm4_pool': True},
    {**V5_BASE, **V5_FULL, 'name': 'V5-A2-NoM2', 'm2_cascade': False, 'm4_pool': True},
    {**V5_BASE, **V5_FULL, 'name': 'V5-A3-NoNotes', 'm3_tool_notes': False, 'm4_pool': True},
    {**V5_BASE, **V5_FULL, 'name': 'V5-A4-NoM4'},                       # no pooling
    {**V5_BASE, **V5_FULL, 'name': 'V5-A5-NoM5', 'm5_strategy': 'refine', 'm4_pool': True},
    {**V5_BASE, 'name': 'V5-A6-OnlyM3', 'm3_structured_reviewer': True,
     'm3_tool_notes': True, 'm3_plan_freeze': True, 'm3_answer_format': True},
    # M4 control: equal-split pooling (no ranking) — isolates the value of ranking
    {**V5_BASE, **V5_FULL, 'name': 'V5-M4-Equal', 'm4_pool_equal': True},
    # v5 upper bound: no mechanisms, cascade off, plain fixed pipeline
    {**V5_BASE, 'name': 'V5-Unconstrained'},
    # v5.2 C1: Unc + ONLY the ANSWER:-line instruction (m3_answer_format touches
    # nothing but the prompt suffix) — decides whether retrieval-task gains are
    # just an extraction advantage (I1).
    {**V5_BASE, 'name': 'V5-Unc-Fmt', 'm3_answer_format': True},
    # ---- v5.1: ONE unified config for all dataset types ----
    # Verbalized escalation (model can decline to direct-answer) + roomier
    # L0/L1 (2048 tok, no CoT truncation). Fixes the MATH cascade failure
    # without giving up the retrieval-task wins. M4 pooling stays on.
    {**V5_BASE, **V5_FULL, 'name': 'V5.1-Unified', 'm2_verbal': True,
     'l01_max_tokens': 2048, 'm4_pool': True},
    {**V5_BASE, **V5_FULL, 'name': 'V5.1-NoPool', 'm2_verbal': True,
     'l01_max_tokens': 2048},
    # Ab7: identical to V5.1-NoPool except the model may no longer declare
    # ESCALATE, so the ladder is routed by the log-probability gate alone. This
    # is the arm that isolates the verbalized channel; every other lever, the
    # bounded slots, M1 early stop, the L0/L1 budget, is held fixed.
    {**V5_BASE, **V5_FULL, 'name': 'V5.1-Ab7-NoVerbal', 'm2_verbal': False,
     'l01_max_tokens': 2048},
    # Reasoning as a rung. On a thinking model the carried-slot bounds do not touch
    # the term that actually grows, which is the per-call generation: measured on
    # GAIA at 24k the governed arm spent 10,998 tokens against the baseline's
    # 11,229, a difference of -2% where the same arm saves 24-67% on models whose
    # thinking is off. This arm gives the levels the gate keeps cheap a cheaper
    # reasoning setting and leaves everything else identical, so the comparison
    # against V5.1-Ab7-NoVerbal prices that one change. On MATH, reasoning_effort
    # low costs 167 generated tokens against 566 at the default and 65.0% against
    # 76.7% accuracy over 60 items, which is the trade the ladder is meant to make
    # only for the items the gate is confident about.
    {**V5_BASE, **V5_FULL, 'name': 'V5.2-ReasonLadder', 'm2_verbal': False,
     'l01_max_tokens': 2048, 'l01_reasoning_effort': 'low',
     'l01_thinking_budget': int(os.getenv('L01_THINKING_BUDGET', '0')) or None,
     'l01_thinking_off': os.getenv('L01_THINKING_OFF', '') == '1'},
    # Reasoning governed by the allowance rather than pinned to the level. ReasonLadder
    # gives the cheap levels a cheap constant; this arm instead lets every call spend a
    # fraction of what is left, so the reasoning term shrinks as the budget is consumed
    # the way every other term already does. The pair against ReasonLadder prices the
    # difference between a fixed schedule and an actual budget coupling.
    {**V5_BASE, **V5_FULL, 'name': 'V5.3-ReasonBudget', 'm2_verbal': False,
     'l01_max_tokens': 2048, 'l01_reasoning_effort': 'low',
     'reason_budget_frac': 0.15, 'reason_budget_min': 128, 'reason_budget_max': 4096},
    # ---- v5.2 Phase Ab: v5.1's own ablations (GAIA+MATH @24k) ----
    # Ab3: drop M2 cascade entirely (fixed L2 pipeline + M1 + M3) — highest
    # priority: does the cascade earn its keep beyond MATH?
    {**V5_BASE, **V5_FULL, 'name': 'V5.1-Ab3-NoM2', 'm2_cascade': False},
    # Ab1: drop M1 early stop under the verbalized router
    {**V5_BASE, **V5_FULL, 'name': 'V5.1-Ab1-NoM1', 'm1_early_stop': False,
     'm2_verbal': True, 'l01_max_tokens': 2048},
    # Ab2: drop M3.2 tool notes (keep raw tool output)
    {**V5_BASE, **V5_FULL, 'name': 'V5.1-Ab2-NoNotes', 'm3_tool_notes': False,
     'm2_verbal': True, 'l01_max_tokens': 2048},
    # v5.3 E2: drop M3 context-economics part (reviewer JSON + tool notes +
    # plan freeze) but KEEP m3_answer_format — isolates M3's SR contribution
    # on MuSiQue from the format-alignment effect (Z1 ruled out M1 there).
    {**V5_BASE, **V5_FULL, 'name': 'V5.1-Ab5-NoM3ctx',
     'm3_structured_reviewer': False, 'm3_tool_notes': False,
     'm3_plan_freeze': False, 'm2_verbal': True, 'l01_max_tokens': 2048},
    # ---- v5.3 Z1b follow-up: Ab6 = Ab3 minus M1 (fixed L2 pipeline + M3 only) ----
    # Why this arm exists: Ab1 (-M1 under the cascade) is a NO-OP on MATH — the
    # verbalized router parks 305/400 tasks at L0 (single round), where M1 has
    # nothing to stop, so Ab1 cannot test M1 at all. Only with the cascade OFF
    # does every task run the full multi-round loop and M1 become able to act.
    #   Ab3 (fixed pipe + M1 + M3) = 70.8%   <-- measured
    #   Ab6 (fixed pipe +      M3) = ?        <-- this arm
    # Ab3 - Ab6 is the first clean estimate of M1's own SR contribution.
    {**V5_BASE, **V5_FULL, 'name': 'V5.1-Ab6-NoM2NoM1',
     'm2_cascade': False, 'm1_early_stop': False},
    # ---- Compaction baseline (the dominant approach in recent agent scaffolds) ----
    # The naive fixed pipeline, but the executor context ACCUMULATES the full
    # trajectory and is rewritten by an LLM summarizer once it crosses the
    # trigger. This is the accumulate-then-compact regime; our system instead
    # caps each carried slot so nothing large enough to need compaction is ever
    # built. The summarizer's own tokens are charged to the task budget.
    # NOTE: the pre-existing 'Unc+Compact' arm is NOT this -- it drops prior
    # turns behind a marker without calling a model, i.e. truncation. Do not
    # report that one as a compaction baseline.
    # Accumulate WITHOUT summarizing. The compaction arm changes two things at
    # once, and its numbers cannot separate them: it accumulates the trajectory
    # AND summarizes past a threshold. On FRAMES the summarizer fired on six
    # items while truncation went from 255 to 455, so almost all of that harm is
    # the accumulation, not the compaction. This arm isolates it.
    {**V5_BASE, 'name': 'V5-Accum-Only', 'm3_answer_format': True,
     'accumulate_context': True},

    # ---- BigCodeBench: library-heavy synthesis, execution-judged ----
    # m3_answer_format stays OFF: the answer is a fenced code block and an
    # appended "ANSWER:" line fights it. The output format is instead carried by
    # the task prompt, identically for every arm.
    {**V5_BASE, 'name': 'BCB-Naive'},
    {**V5_BASE, **V5_FULL, 'name': 'BCB-Ours', 'm2_verbal': True,
     'l01_max_tokens': 2048, 'm3_answer_format': False},
    {**V5_BASE, **V5_FULL, 'name': 'BCB-Ab7', 'm2_verbal': False,
     'l01_max_tokens': 2048, 'm3_answer_format': False},
    # BigCodeBench on a model whose thinking cannot be turned off. Same relation to
    # BCB-Ab7 as V5.2-ReasonLadder has to V5.1-Ab7-NoVerbal: the levels the gate keeps get
    # a cheaper reasoning setting, everything else is identical.
    {**V5_BASE, **V5_FULL, 'name': 'BCB-ReasonLadder', 'm2_verbal': False,
     'l01_max_tokens': 2048, 'm3_answer_format': False, 'l01_reasoning_effort': 'low',
     'l01_thinking_budget': int(os.getenv('L01_THINKING_BUDGET', '0')) or None,
     'l01_thinking_off': os.getenv('L01_THINKING_OFF', '') == '1'},

    # ---- Single-call baselines and the all-escalate counterfactual (analysis paper) ----
    # One Executor call with the format instruction, no plan, no tool, no reviewer, default
    # reasoning. Run at two generation caps because the pipeline baselines call the Executor
    # with MAX_TOKENS=1024 while the governed L0/L1 calls get 2048, and on a reasoning model
    # that difference alone decides how many answers come back empty. 1k matches the
    # baselines' per-call cap, 2k matches the governed arm's cheap rungs.
    {**V5_BASE, 'name': 'V5-Single-Fmt-1k', 'm2_cascade': True, 'm3_answer_format': True,
     'v5_max_level': 0, 'l0_no_tool': True, 'l01_max_tokens': 1024, 'm5_strategy': 'refine'},
    {**V5_BASE, 'name': 'V5-Single-Fmt-2k', 'm2_cascade': True, 'm3_answer_format': True,
     'v5_max_level': 0, 'l0_no_tool': True, 'l01_max_tokens': 2048, 'm5_strategy': 'refine'},
    # Same single call at the 4096 cap the pipeline comparisons use, so that the single-call
    # and pipeline arms are compared at an equal per-call cap.
    {**V5_BASE, 'name': 'V5-Single-Fmt-4k', 'm2_cascade': True, 'm3_answer_format': True,
     'v5_max_level': 0, 'l0_no_tool': True, 'l01_max_tokens': 4096, 'm5_strategy': 'refine'},
    # Review N3: the single call with its generation cap raised to 16k, to separate longer
    # reasoning from multi-agent coordination under the same 24k budget. (On the GLM deployment
    # the default reasoning setting already reasons at least as much as reasoning_effort=high,
    # so the 4k cap, not the setting, is what limits the single call.)
    {**V5_BASE, 'name': 'V5-Single-Fmt-16k', 'm2_cascade': True, 'm3_answer_format': True,
     'v5_max_level': 0, 'l0_no_tool': True, 'l01_max_tokens': 16384, 'm5_strategy': 'refine'},
    {**V5_BASE, 'name': 'BCB-Single-4k', 'm2_cascade': True, 'm3_answer_format': False,
     'v5_max_level': 0, 'l0_no_tool': True, 'l01_max_tokens': 4096, 'm5_strategy': 'refine'},
    {**V5_BASE, 'name': 'BCB-Single-1k', 'm2_cascade': True, 'm3_answer_format': False,
     'v5_max_level': 0, 'l0_no_tool': True, 'l01_max_tokens': 1024, 'm5_strategy': 'refine'},
    {**V5_BASE, 'name': 'BCB-Single-2k', 'm2_cascade': True, 'm3_answer_format': False,
     'v5_max_level': 0, 'l0_no_tool': True, 'l01_max_tokens': 2048, 'm5_strategy': 'refine'},
    # The governed arms with gates that never pass, so every task runs L0, L1 and L2 and the
    # trace holds all three rungs' outcomes per item. Everything else is the governed arm as
    # run, which is what makes the per-rung counterfactual and the oracle router computable.
    {**V5_BASE, **V5_FULL, 'name': 'V5.1-Ab7-AllEsc', 'm2_verbal': False,
     'l01_max_tokens': 2048, 'theta0': 99.0, 'theta1': 99.0},
    {**V5_BASE, **V5_FULL, 'name': 'V5.2-ReasonLadder-AllEsc', 'm2_verbal': False,
     'l01_max_tokens': 2048, 'l01_reasoning_effort': 'low', 'theta0': 99.0, 'theta1': 99.0,
     'l01_thinking_budget': int(os.getenv('L01_THINKING_BUDGET', '0')) or None,
     'l01_thinking_off': os.getenv('L01_THINKING_OFF', '') == '1'},
    {**V5_BASE, **V5_FULL, 'name': 'BCB-ReasonLadder-AllEsc', 'm2_verbal': False,
     'l01_max_tokens': 2048, 'm3_answer_format': False, 'l01_reasoning_effort': 'low',
     'theta0': 99.0, 'theta1': 99.0,
     'l01_thinking_budget': int(os.getenv('L01_THINKING_BUDGET', '0')) or None,
     'l01_thinking_off': os.getenv('L01_THINKING_OFF', '') == '1'},

    # ---- State-policy comparison (price vs value of carried state) ----
    # All start at the full loop (no ladder), carry the ANSWER: instruction, and share the
    # stop rule: stop when the reviewer accepts, or at round 5. Only the state carried across
    # rounds differs. ST-Rebuild is V5-Unc-Fmt (rebuild from the last turn, full plan, free-form
    # critique). The two -M1 arms swap in the strict log-probability stop to separate
    # over-revision from information loss.
    {**V5_BASE, 'name': 'ST-Accum', 'm3_answer_format': True, 'state_policy': 'accumulate'},
    {**V5_BASE, 'name': 'ST-Window2', 'm3_answer_format': True, 'state_policy': 'window', 'state_window': 2},
    {**V5_BASE, 'name': 'ST-Summary', 'm3_answer_format': True, 'state_policy': 'summary', 'state_summary_tokens': 256},
    {**V5_BASE, 'name': 'ST-Slots', 'm3_answer_format': True, 'm3_plan_freeze': True,
     'm3_structured_reviewer': True, 'm3_tool_notes': True},
    {**V5_BASE, 'name': 'ST-Slots-M1', 'm3_answer_format': True, 'm3_plan_freeze': True,
     'm3_structured_reviewer': True, 'm3_tool_notes': True, 'm1_early_stop': True},
    {**V5_BASE, 'name': 'ST-Rebuild-M1', 'm3_answer_format': True, 'm1_early_stop': True},
    # Single-component ablations of ST-Slots: which of its three changes discards what
    # revision needs. Each turns on one slot cap and leaves the rest as in rebuild.
    {**V5_BASE, 'name': 'ST-PlanOnly', 'm3_answer_format': True, 'm3_plan_freeze': True},
    {**V5_BASE, 'name': 'ST-CritOnly', 'm3_answer_format': True, 'm3_structured_reviewer': True},
    {**V5_BASE, 'name': 'ST-ToolOnly', 'm3_answer_format': True, 'm3_tool_notes': True},
    # The structured reviewer's generation is capped at 150 tokens. On a reasoning model that
    # can truncate the review before any verdict is written, and an empty review is read as
    # "revise". These two arms raise only that cap, to separate the critique's format from its
    # truncation.
    {**V5_BASE, 'name': 'ST-CritOnly-R1k', 'm3_answer_format': True, 'm3_structured_reviewer': True,
     'review_max_tokens': 1024},
    {**V5_BASE, 'name': 'ST-Slots-R1k', 'm3_answer_format': True, 'm3_plan_freeze': True,
     'm3_structured_reviewer': True, 'm3_tool_notes': True, 'review_max_tokens': 1024},
    # Reviewer cap equal to every other call's (4096), to separate the critique format's own
    # effect from what remains of truncation at 1024 on a model that reasons longer.
    {**V5_BASE, 'name': 'ST-CritOnly-R4k', 'm3_answer_format': True, 'm3_structured_reviewer': True,
     'review_max_tokens': 4096},
    {**V5_BASE, 'name': 'ST-Slots-R4k', 'm3_answer_format': True, 'm3_plan_freeze': True,
     'm3_structured_reviewer': True, 'm3_tool_notes': True, 'review_max_tokens': 4096},

    # ---- Runtime per-call cap (feasibility test) ----
    # The plain pipeline (Naive+Fmt) with the per-call cap set at run time from the remaining
    # budget, and an empty review read as "keep". RC-CapOnly isolates the cap rule.
    {**V5_BASE, 'name': 'RC-Adaptive', 'm3_answer_format': True, 'runtime_cap': True,
     'runtime_cap_k': 3, 'runtime_cap_lo': 512, 'runtime_cap_hi': 4096, 'empty_review_keep': True},
    {**V5_BASE, 'name': 'RC-CapOnly', 'm3_answer_format': True, 'runtime_cap': True,
     'runtime_cap_k': 3, 'runtime_cap_lo': 512, 'runtime_cap_hi': 4096},
    # ZEBRA-style static allocation: per-role shares fixed before execution, from measured role
    # costs (GLM 0.21/0.48/0.31, DeepSeek 0.30/0.42/0.28), two expected rounds.
    {**V5_BASE, 'name': 'SS-Static', 'm3_answer_format': True,
     'static_split': {'planner': float(os.getenv('SS_P', '0.25')), 'executor': float(os.getenv('SS_E', '0.45')),
                      'reviewer': float(os.getenv('SS_R', '0.30'))},
     'static_calls': {'planner': 1, 'executor': 2, 'reviewer': 2}},
    {**V5_BASE, 'name': 'SS-Static-Keep', 'm3_answer_format': True, 'empty_review_keep': True,
     'static_split': {'planner': float(os.getenv('SS_P', '0.25')), 'executor': float(os.getenv('SS_E', '0.45')),
                      'reviewer': float(os.getenv('SS_R', '0.30'))},
     'static_calls': {'planner': 1, 'executor': 2, 'reviewer': 2}},

    # ---- CLIMB, live: L0 (cheap reasoning, 2,048 cap) -> L1 (tool, seeded) -> the plain
    # pipeline as L2 (full plan, free-form reviewer, stop on accept), seeded with the draft.
    # Thresholds certified offline on a disjoint calibration half, passed as THETA_LIVE.
    # BigCodeBench variant: identical except that the answer is the code block itself, so no
    # 'ANSWER:' line is requested (the reference there is BCB-Naive, V5-Unc-Fmt without that line).
    {**V5_BASE, 'name': 'CLIMB-Live-BCB', 'm2_cascade': True, 'm3_answer_format': False,
     'l01_max_tokens': 2048, 'l01_reasoning_effort': 'low',
     'l01_thinking_off': os.getenv('L01_THINKING_OFF', '') == '1',
     'theta0': float(os.getenv('THETA_LIVE', '99')), 'theta1': float(os.getenv('THETA_LIVE', '99'))},
    {**V5_BASE, 'name': 'CLIMB-Live', 'm2_cascade': True, 'm3_answer_format': True,
     'l01_max_tokens': 2048, 'l01_reasoning_effort': 'low',
     'l01_thinking_off': os.getenv('L01_THINKING_OFF', '') == '1',
     'theta0': float(os.getenv('THETA_LIVE', '99')), 'theta1': float(os.getenv('THETA_LIVE', '99'))},

    # ---- SWE-bench Verified, oracle localization ----
    # The answer format is carried by the task prompt itself (labelled REGION
    # blocks), identically for every arm, so m3_answer_format stays OFF here:
    # turning it on would append an "ANSWER:" line that fights the code blocks.
    {**V5_BASE, 'name': 'SWEO-Naive'},
    {**V5_BASE, **V5_FULL, 'name': 'SWEO-Ours', 'm2_verbal': True,
     'l01_max_tokens': 2048, 'm3_answer_format': False},
    # The Ab7 arm needs its own SWEO entry for the same reason SWEO-Ours does.
    # Reusing the general-purpose V5.1-Ab7-NoVerbal inherits m3_answer_format
    # from V5_FULL, which appends "End your reply with a single line: ANSWER: x"
    # to a task whose answer is a set of fenced code regions. That instruction
    # appeared in 421 of 500 replies and cut REGION-format compliance from ~300
    # to 195, so the first SWE run of this arm measured the conflict, not the arm.
    {**V5_BASE, **V5_FULL, 'name': 'SWEO-Ab7', 'm2_verbal': False,
     'l01_max_tokens': 2048, 'm3_answer_format': False},

    # ---- E4: on-policy single-slot bounding (motivation, section 1) ----
    # Each arm is the naive format-aligned baseline with EXACTLY ONE carried
    # slot bounded, run end to end. This is the on-policy counterpart of the
    # counterfactual slot ablation: there, a slot was swapped at one round of a
    # trajectory the unbounded pipeline had produced, so the planner had written
    # its plan to be read whole. Here the pipeline runs under the bound from the
    # first call, which is the only way to price a slot without that confound.
    {**V5_BASE, 'name': 'E4-BoundPlan', 'm3_answer_format': True, 'm3_plan_freeze': True},
    {**V5_BASE, 'name': 'E4-BoundCritique', 'm3_answer_format': True, 'm3_structured_reviewer': True},
    {**V5_BASE, 'name': 'E4-BoundNotes', 'm3_answer_format': True, 'm3_tool_notes': True},

    # Compaction baselines. `kv_trigger` is now the fraction of `context_window`
    # the prompt must occupy before the summarizer runs (it used to be a fraction of
    # the *budget already spent*, which made the arm unreadable; see should_summarize).
    # Realistic: an 8k window never fires on our tasks (measured peak prompt 4,067).
    # Forced: a 2k window makes compaction operate, to price it when it does run.
    {**V5_BASE, 'name': 'V5-Compact-Real', 'm3_answer_format': True,
     'accumulate_context': True, 'enable_kv': True, 'kv_trigger': 0.5,
     'context_window': 8192, 'real_summary': True, 'summary_budget': 512},
    {**V5_BASE, 'name': 'V5-Compact-Forced', 'm3_answer_format': True,
     'accumulate_context': True, 'enable_kv': True, 'kv_trigger': 0.5,
     'context_window': 2048, 'real_summary': True, 'summary_budget': 512},
    # Conservative trigger. At 0.5 the probe compacted on nearly every round
    # (20 firings over 8 tasks that ran 1.8 rounds each), which is harsher than
    # a real scaffold would be; reporting only that setting would invite the
    # charge that we crippled the baseline. 0.8 fires late, so the pair brackets
    # the compaction-frequency axis rather than picking one point on it.
    {**V5_BASE, 'name': 'V5-Compact-Late', 'm3_answer_format': True,
     'accumulate_context': True, 'enable_kv': True, 'kv_trigger': 0.8,
     'real_summary': True, 'summary_budget': 512},
]
if os.getenv('V5', '0') == '1':
    CONFIGS = CONFIGS_V5
# Calibration mode (V5_CALIB=1): observe every level/round per task — no early
# stop, thetas at +inf so nothing is stopped by the cascade, no pooling.
# Offline threshold sweeps then replay these traces (runs are temp=0, ~deterministic).
if os.getenv('V5_CALIB', '0') == '1':
    for c in CONFIGS:
        c.update(m1_early_stop=False, theta0=1e9, theta1=1e9, m5_strategy='refine')
        c.pop('m4_pool', None)
        c.pop('m4_pool_equal', None)


# Optional single-config / module-ablation overrides (for tight-budget ablations)
_only = os.getenv('CONFIG_ONLY')
if _only:
    _names = set(_only.split(','))
    CONFIGS = [c for c in CONFIGS if c['name'] in _names]
# v5 Phase 0 diagnostic runs: capture per-token logprob/entropy on every call
if os.getenv('CAPTURE_LOGPROBS', '0') == '1':
    for c in CONFIGS:
        c['capture_logprobs'] = True
if os.getenv('DISABLE_BORROWING', '0') == '1':
    for c in CONFIGS:
        c['enable_borrowing'] = False
        c['name'] = c['name'] + '-NoBorrow'
if os.getenv('ENABLE_CREDIT_SCORING', '0') == '1':
    for c in CONFIGS:
        if c.get('enable_borrowing'):
            c['enable_credit_scoring'] = True
            c['name'] = c['name'] + '+Credit'

print(f'Budget-constrained sweep on GAIA ({MODEL_TAG}): {len(TASKS)} tasks x '
      f'{len(CONFIGS)} configs x {len(BUDGETS_SWEEP)} budgets x {len(SEEDS)} seeds')


def dump_per_task(r, seed, budget, name, fh):
    for tr in r['results']:
        bs = tr['bank_summary']
        fh.write(json.dumps({
            'seed': seed, 'budget': budget, 'config': name, 'task_id': tr['task_id'],
            'success': tr['success'], 'total_used': bs['total_used'],
            'cost_usd': bs['total_cost_usd'], 'borrows': bs['borrow_count'],
            'stalled': tr.get('stalled', False),
            'n_blocked': sum(1 for c in tr.get('agent_calls', []) if not c['success']),
            # v5 instrumentation (Appendix B)
            'f1': tr.get('f1', 0.0),
            'rounds': tr.get('rounds', []),
            'stopped_by': tr.get('stopped_by'),
            'wall_clock_ms': tr.get('wall_clock_ms'),
            'flags_fired': tr.get('flags_fired', {}),
            'escalation_level': tr.get('escalation_level'),
            # per-role spend; `rounds` holds executor turns only, so this is
            # the only record of what planner and reviewer cost
            'role_tokens': tr.get('role_tokens', {}),
            'segment_chars': tr.get('segment_chars', {}),
            'fixed_prompt_tokens': tr.get('fixed_prompt_tokens'),
            'raw_answer': tr.get('raw_answer', ''),
        }) + '\n')
    fh.flush()


_tag = os.getenv('OUT_TAG', '')
# Traces land in traces/ rather than the working directory. They used to be written beside
# the code, which is how 85 of them came to sit at the top level alongside the scripts while
# an earlier family's runs sat in traces/, and a reader of the file map had to know which
# convention each family followed. TRACE_DIR overrides it for a throwaway run.
_trace_dir = os.getenv('TRACE_DIR', 'traces')
os.makedirs(_trace_dir, exist_ok=True)
_pt_fh = open(os.path.join(_trace_dir, f'tasks_sweep_{MODEL_TAG}{_tag}.jsonl'), 'a')


def run_m4_pool(tasks, runner, cfg, llm, budget, ranked=True):
    """v5 M4: dataset-level pooled budget, two passes.
    Pass 1: every task probes at L0/L1 with b0 = 0.3*B. Converged tasks finish.
    Ranking: remaining pool goes to unfinished tasks by marginal value
    v_i = (1 - conf_i) / c_i (conf from pass-1 logprob, c_i = P50 probe cost).
    ranked=False is the equal-split control (isolates the value of ranking)."""
    n = len(tasks)
    b0 = max(1000, int(0.3 * budget))
    c1 = dict(cfg); c1['budget'] = b0; c1['min_quota'] = max(1, b0 // 5)
    c1['v5_max_level'] = 1
    print(f'  [M4] pass 1: probe {n} tasks @ b0={b0}')
    r1 = run_experiment_v2(tasks, runner, c1, llm)
    used1 = sum(t['bank_summary']['total_used'] for t in r1['results'])
    pool = budget * n - used1
    finals, pending = {}, []
    for t, tr in zip(tasks, r1['results']):
        lp = None
        for rnd in reversed(tr.get('rounds', [])):
            if rnd.get('mean_logprob') is not None:
                lp = rnd['mean_logprob']; break
        conf = min(1.0, max(0.0, ((lp if lp is not None else -0.3) + 0.3) / 0.3))
        # v5.1 fix: a verbalized ESCALATE is a non-answer — its logprob is sky
        # high (4 confident tokens), which fooled the ranker into dropping the
        # task with "ESCALATE" as its final answer. Force conf=0 so it always
        # stays pending for pass 2.
        if (tr.get('answer') or '').strip().upper().startswith('ESCALATE'):
            conf = 0.0
        if tr.get('stopped_by') == 'converged':
            finals[t['id']] = tr
        else:
            pending.append((t, tr, conf))
    costs = [tr['bank_summary']['total_used'] for _, tr, _ in pending]
    c_i = float(np.median(costs)) if costs else 1.0
    print(f'  [M4] pass 1 done: {len(finals)} converged, {len(pending)} pending, '
          f'pool={pool}')
    winners = []
    if ranked:
        pending.sort(key=lambda x: -(1 - x[2]) / max(c_i, 1))
        for t, tr, conf in pending:
            if pool >= budget and (1 - conf) > 0.05:
                winners.append((t, tr)); pool -= budget
            else:
                finals[t['id']] = tr
    else:
        share = pool // max(len(pending), 1)
        for t, tr, conf in pending:
            if share >= 2000:
                winners.append((t, tr))
            else:
                finals[t['id']] = tr
    if winners:
        pass2_tasks = []
        for t, tr in winners:
            t2 = dict(t)
            t2['_v5_start_level'] = 2
            t2['_v5_draft'] = tr.get('answer') or ''
            pass2_tasks.append(t2)
        c2 = dict(cfg); c2['budget'] = budget
        print(f'  [M4] pass 2: escalate {len(pass2_tasks)} tasks to L2 @ {budget}')
        r2 = run_experiment_v2(pass2_tasks, runner, c2, llm)
        for t2, tr1, tr2 in zip(pass2_tasks, [w[1] for w in winners], r2['results']):
            # merge pass-1 spend into the pass-2 record
            u1 = tr1['bank_summary']['total_used']
            tr2['bank_summary']['total_used'] += u1
            tr2['rounds'] = tr1.get('rounds', []) + tr2.get('rounds', [])
            finals[t2['id']] = tr2
    results = [finals[t['id']] for t in tasks]
    succ = [r['success'] for r in results]
    toks = [r['bank_summary']['total_used'] for r in results]
    cost = sum(r['bank_summary']['total_cost_usd'] for r in results)
    return {'config': cfg, 'results': results, 'success_rate': float(np.mean(succ)),
            'mean_tokens': float(np.mean(toks)), 'std_tokens': float(np.std(toks)),
            'overrun_rate': 0.0, 'recovery_rate': 0.0, 'total_cost_usd': cost,
            'cost_per_success': cost / (sum(succ) or 1), 'mean_borrows': 0.0,
            'opl_tasks': 0, 'opl_learned': {}, 'credit_scores': {}}


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
            print(f"\n--- {c['name']} @ budget={budget} seed={seed} ---")
            if c.get('m4_pool') or c.get('m4_pool_equal'):
                r = run_m4_pool(TASKS, RUNNER, c, llm, c['budget'],
                                ranked=bool(c.get('m4_pool')))
            else:
                r = run_experiment_v2(TASKS, RUNNER, c, llm)
            dump_per_task(r, seed, budget, c['name'], _pt_fh)
            sweep.append({'seed': seed, 'budget': budget, 'name': c['name'],
                          'success_rate': r['success_rate'], 'cps': r['cost_per_success'],
                          'mean_tokens': r['mean_tokens'], 'cost': r['total_cost_usd']})
            print(f"  SR={r['success_rate']:.1%} CPS=${r['cost_per_success']:.4f} "
                  f"tokens={r['mean_tokens']:.0f}")

rows = []
for budget in BUDGETS_SWEEP:
    for name in [c['name'] for c in CONFIGS]:
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

out_json = os.path.join(os.getenv('SUMMARY_DIR', 'summaries'),
                        f'budget_sweep_v2_{MODEL_TAG}{_tag}.json')
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, 'w') as f:
    json.dump({'model': PRIMARY_MODEL, 'seeds': SEEDS, 'budgets': BUDGETS_SWEEP,
               'raw': sweep, 'agg': rows}, f, indent=2)
print(f'\nExported: {out_json}')
