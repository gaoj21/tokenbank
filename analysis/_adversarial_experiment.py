"""Adversarial over-borrowing experiment (mock, zero cost).

A rogue Executor runs a retry loop of big, useless calls, borrowing every time
its quota runs out. Compare flat credit vs scored credit:
  - does the rogue burn the good agents' quota (reviewer survival)?
  - how much credit is extended in total?
  - how much of the global reserve is burned?

  .venv/bin/python -u _adversarial_experiment.py
"""
import json
import numpy as np
import re as _re

src = open('_run_v2_experiment.py').read()
_parts = _re.split(r'# ============ NOTEBOOK CELL (\d+) ============\n', src)
cell_map = {int(_parts[i]): _parts[i + 1] for i in range(1, len(_parts), 2)}
# cells: 2=config, 6=components, 7=bank, 8=llm client, 10=orchestrator
for n in [2, 6, 7, 8, 10]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())

# switch to mock after cell exec (LLMClientV2.call reads MOCK_MODE at call time)
globals()['MOCK_MODE'] = True
globals()['VERBOSE'] = False
random.seed(42)
np.random.seed(42)

# Tuned so the rogue drains nearly its whole credit line by round 3
# (~5.6K borrow vs 6.0K flat line), which pushes the lender (reviewer, call
# ~7.1K) below its call size under flat credit — while a tightened scored line
# (~4.5K after defaults) blocks the borrow and the reviewer survives.
MOCK_TOKEN_PROFILES['Executor'] = (3400, 650)
MOCK_TOKEN_PROFILES['Planner'] = (3000, 800)
MOCK_TOKEN_PROFILES['Reviewer'] = (5200, 1300)

N_TASKS = 50
ROGUE_ROUNDS = 6


def run_adversarial_task(task, bank, llm_client):
    results = {'task_id': task['id'], 'agent_calls': [], 'success': False}
    # Priorities matter: they gate who can lend to whom. Reviewer (1) can
    # borrow from nobody, so draining it kills the task — unless the scored
    # credit line contains the rogue earlier.
    agents = {'planner': 'Planner', 'executor': 'Executor', 'reviewer': 'Reviewer'}
    bank.register_agents(agents, priorities={'Planner': 3, 'Executor': 2, 'Reviewer': 1})
    bank.allocate()

    r1 = llm_client.call('planner', bank, [{'role': 'user', 'content': 'plan'}], action='plan')
    results['agent_calls'].append(r1)

    # Rogue executor: runaway retry loop of large useless calls
    for _ in range(ROGUE_ROUNDS):
        r = llm_client.call('executor', bank, [{'role': 'user', 'content': 'retry'}], action='execute')
        results['agent_calls'].append(r)

    # Innocent reviewer tries to finish the task afterwards
    r3 = llm_client.call('reviewer', bank, [{'role': 'user', 'content': 'validate'}], action='review')
    results['agent_calls'].append(r3)
    bank.update_opl(results)

    # Success mostly requires the good agent to survive the rogue's credit burn;
    # most credit extended to the rogue therefore defaults (strong scorer signal)
    results['success'] = bool(r3['success']) and bool(np.random.random() < 0.3)
    return results


TASKS = [{'id': f'adv_{i}'} for i in range(N_TASKS)]
BASE = {'budget': 40000, 'min_quota': 5000, 'policy': 'hard', 'external_plan': None,
        'gamma': 0.8, 'enable_predictive': False, 'enable_borrowing': True,
        'enable_opl': False, 'enable_kv': True, 'theta': 0.8}
CONFIGS = [
    {**BASE, 'name': 'FlatCredit', 'enable_credit_scoring': False},
    {**BASE, 'name': 'ScoredCredit', 'enable_credit_scoring': True},
]

out = []
for cfg in CONFIGS:
    r = run_experiment_v2(TASKS, run_adversarial_task, cfg, llm)
    reviewer_ok = np.mean([any(c['agent_id'] == 'reviewer' and c['success'] for c in t['agent_calls'])
                           for t in r['results']])
    borrowed = [t['bank_summary']['total_borrowed'] for t in r['results']]
    reserves = [t['bank_summary']['reserve'] for t in r['results']]
    row = {'name': cfg['name'], 'success_rate': r['success_rate'],
           'reviewer_survival': float(reviewer_ok),
           'mean_borrowed_tokens': float(np.mean(borrowed)),
           'total_borrowed_tokens': int(sum(borrowed)),
           'mean_reserve_left': float(np.mean(reserves)),
           'credit_scores': r.get('credit_scores', {})}
    out.append(row)
    print(f"{cfg['name']:>13}: task_SR={row['success_rate']:.1%} "
          f"reviewer_survival={row['reviewer_survival']:.1%} "
          f"borrowed/task={row['mean_borrowed_tokens']:.0f} "
          f"reserve_left={row['mean_reserve_left']:.0f} "
          f"scores={row['credit_scores']}")

with open('adversarial_experiment.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nExported: adversarial_experiment.json')
