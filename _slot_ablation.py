#!/usr/bin/env python
"""Counterfactual slot ablation: do the tokens we drop carry any information?

The paper's evidence for capping the carried slots is indirect -- cap them and
accuracy rises. This measures the claim directly. On a live round of the naive
(unbounded) pipeline we re-issue the SAME executor call several times, each time
replacing exactly one carried slot with our bounded version, and record whether
the answer moves and whether it moves toward or away from correct.

Reads: P(answer unchanged) is inertness -- the slot's tokens bought nothing.
Among the answers that do move, correct->wrong versus wrong->correct says
whether the slot was merely inert or actively harmful.

Usage:
  OPENAI_BASE_URL=... VLLM_API_KEY=... VLLM_MODEL=... BENCHMARK=MATH N=150 \
  .venv/bin/python -u _slot_ablation.py
"""
import os, json, re, random, collections
import concurrent.futures as cf

import re as _re
src = open('_run_v2_experiment.py').read()
_parts = _re.split(r'# ============ NOTEBOOK CELL (\d+) ============\n', src)
cell_map = {int(_parts[i]): _parts[i + 1] for i in range(1, len(_parts), 2)}
for n in [2]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())
random.seed(42)
import numpy as np; np.random.seed(42)
for n in [4, 6, 7, 8, 9]:
    exec(compile(cell_map[n], f'cell{n}', 'exec'), globals())

from openai import OpenAI
CLIENT = OpenAI(api_key=os.getenv('VLLM_API_KEY', 'EMPTY'),
                base_url=os.getenv('OPENAI_BASE_URL'), timeout=180, max_retries=2)
MODEL = os.getenv('VLLM_MODEL')
EB = {'chat_template_kwargs': {'enable_thinking': False}}
ANS = ' End your reply with a single line: "ANSWER: <final answer>".'


def ask(messages, max_tokens=1024):
    r = CLIENT.chat.completions.create(model=MODEL, messages=messages, temperature=0,
                                       max_tokens=max_tokens, extra_body=EB)
    return r.choices[0].message.content or '', r.usage.completion_tokens


def build_tasks():
    b = os.getenv('BENCHMARK', 'MATH')
    n = int(os.getenv('N', '150'))
    if b == 'MATH':
        from datasets import load_dataset
        ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
        t = [{'id': f'math_{i}', 'question': it['problem'], 'answer': str(it['answer'])}
             for i, it in enumerate(ds)]
    elif b == 'FRAMES':
        t = [{'id': x['id'], 'question': x['question'], 'answer': x['answer']}
             for x in load_frames_dataset(824)]
    elif b == 'SWEO':
        # The 850 paired trials that found the plan slot free are all question
        # answering. On SWE-bench the governed arm produces significantly worse
        # patches than the untreated one (46 against 14 on items where both
        # submit), and the obvious suspect is that a repair plan carries the
        # diagnosis rather than a decomposition. This tests that directly.
        import sys as _s; _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _swe_oracle import render_task
        from datasets import load_dataset
        t = [{'id': it['instance_id'], 'question': render_task(it), 'answer': it['test_patch']}
             for it in load_dataset('SWE-bench/SWE-bench_Verified', split='test')]
    elif b == 'MUSIQUE':
        # loader lives in _budget_sweep_v2.py, which this script does not import;
        # replicated here, id scheme included because judge_task keys off it
        from datasets import load_dataset
        t = [{'id': f"musique_{it['id']}", 'question': it['question'],
              'answer': str(it['answer'])}
             for it in load_dataset('dgslibisey/MuSiQue', split='validation')
             if it.get('answerable', True)]
    else:
        t = [{'id': x['id'], 'question': x['question'], 'answer': x['answer']}
             for x in GAIA_TASKS]
    random.Random(42).shuffle(t)
    return t[:n]


def one_task(task):
    """Run two naive rounds, then ablate each slot on the third call."""
    q = task['question']
    sysmsg = ('You are an Executor. Answer concisely. For ANY computation, put Python code '
              'that prints the result in a ```python block; you will receive the output, '
              'then give the final answer.' + ANS)
    out = {'task_id': task['id'], 'variants': {}}
    try:
        # round 1: plan (full text, as the naive pipeline carries it) + answer
        plan, _ = ask([{'role': 'system', 'content': 'You are a Planner. Break down the question into sub-tasks.'},
                       {'role': 'user', 'content': f'Question: {q}\nProvide a step-by-step plan.'}])
        a1, _ = ask([{'role': 'system', 'content': sysmsg},
                     {'role': 'user', 'content': f'Question: {q}\nPlan: {plan}\nFinal answer.'}])
        # reviewer, free prose (the naive slot)
        crit, _ = ask([{'role': 'system', 'content': 'You are a Reviewer. Validate the answer.'},
                       {'role': 'user', 'content': f'Question: {q}\nProposed: {a1}\nCorrect? Explain.'}])
        # bounded counterparts, exactly what our system would carry
        verdict, _ = ask([{'role': 'system', 'content': 'Reply ONLY with JSON: {"verdict":"accept"|"revise",'
                                                        '"issues":["<=3 items, <=20 words"],"hint":"<=30 words"}. No prose.'},
                          {'role': 'user', 'content': f'Question: {q}\nProposed: {a1}'}], max_tokens=150)
        plan_step = next((l.strip() for l in plan.split('\n') if l.strip()), plan[:200])

        base_head = [{'role': 'system', 'content': sysmsg},
                     {'role': 'user', 'content': f'Question: {q}\nPlan: {plan}\nFinal answer.'}]
        step_head = [{'role': 'system', 'content': sysmsg},
                     {'role': 'user', 'content': f'Question: {q}\nCurrent step: {plan_step}\nFinal answer.'}]
        turn = lambda fb: [{'role': 'assistant', 'content': a1},
                           {'role': 'user', 'content': f'Reviewer feedback: {fb}\nRevise and give the final answer.'}]

        out['slot_chars'] = {'question': len(q), 'plan': len(plan), 'plan_step': len(plan_step),
                             'critique': len(crit), 'verdict': len(verdict), 'answer': len(a1)}
        # Bounding a slot without cost says the slot is compressible. It does not say
        # whether the slot is read at all. These three replace the plan with content that
        # cannot help, holding length and shape roughly fixed, so that any difference is
        # attributable to what the plan says rather than to how much of it there is:
        #   shuffled  - the same sentences in a random order, so the text is on-topic but
        #               its structure as a plan is destroyed
        #   foreign   - a real plan written for a different question in the same benchmark
        #   empty     - no plan slot at all, the floor for this comparison
        # If accuracy is flat across control, shuffled and foreign, the executor is not
        # reading the plan, and bounding it is not compression but deletion of dead weight.
        plan_lines = [l for l in plan.split('\n') if l.strip()]
        shuffled = list(plan_lines); random.Random(hash(task['id']) & 0xffff).shuffle(shuffled)
        shuffled = '\n'.join(shuffled)
        foreign = FOREIGN_PLANS.get(task['id'], '')
        shuf_head = [{'role': 'system', 'content': sysmsg},
                     {'role': 'user', 'content': f'Question: {q}\nPlan: {shuffled}\nFinal answer.'}]
        frgn_head = [{'role': 'system', 'content': sysmsg},
                     {'role': 'user', 'content': f'Question: {q}\nPlan: {foreign}\nFinal answer.'}]
        none_head = [{'role': 'system', 'content': sysmsg},
                     {'role': 'user', 'content': f'Question: {q}\nFinal answer.'}]
        variants = {
            'control':       base_head + turn(crit),          # full plan + prose critique
            'bounded_crit':  base_head + turn(verdict),        # prose -> bounded verdict
            'no_crit':       base_head + turn('(none)'),       # critique removed entirely
            'bounded_plan':  step_head + turn(crit),           # full plan -> current step
            'bounded_both':  step_head + turn(verdict),        # both slots bounded (our system)
            'shuffled_plan': shuf_head + turn(crit),           # same sentences, order destroyed
            'foreign_plan':  frgn_head + turn(crit),           # another question's plan
            'no_plan':       none_head + turn(crit),           # no plan slot at all
        }
        for name, msgs in variants.items():
            ans, ntok = ask(msgs)
            ok, f1 = judge_task(task['id'], ans, task['answer'])
            out['variants'][name] = {
                'norm': _norm_basic(_extract_answer(ans)), 'correct': bool(ok), 'f1': f1,
                'prompt_chars': sum(len(m['content']) for m in msgs), 'out_tokens': ntok}
        return out
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
        return out


tasks = build_tasks()
print(f'slot ablation: {len(tasks)} tasks on {os.getenv("BENCHMARK","MATH")}')

# The foreign-plan variant needs a plan written for a different question, so plans are
# produced in a first pass and paired before any variant runs. The pairing is a fixed
# rotation rather than a random draw, so every task both lends and borrows exactly one
# plan and the comparison is not confounded by which questions happened to be sampled.
def _plan_for(task):
    try:
        pl, _ = ask([{'role': 'system', 'content': 'You are a Planner. Break down the question into sub-tasks.'},
                     {'role': 'user', 'content': f"Question: {task['question']}\nProvide a step-by-step plan."}])
        return task['id'], pl
    except Exception:
        return task['id'], ''
print('pass 1: plans for the foreign-plan pairing')
with cf.ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', '32'))) as ex:
    _plans = dict(ex.map(_plan_for, tasks))
_ids = [t['id'] for t in tasks]
FOREIGN_PLANS = {_ids[i]: _plans[_ids[(i + 1) % len(_ids)]] for i in range(len(_ids))}
print(f'  paired {sum(1 for v in FOREIGN_PLANS.values() if v)}/{len(_ids)}')

print('pass 2: variants')
res = []
with cf.ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', '32'))) as ex:
    for i, r in enumerate(ex.map(one_task, tasks), 1):
        res.append(r)
        if i % 25 == 0:
            print(f'  {i}/{len(tasks)}', flush=True)

# default to the benchmark, not a constant: the chain runs four benchmarks in
# sequence and a constant filename makes each one overwrite the last
tag = os.getenv('OUT_TAG') or os.getenv('BENCHMARK', 'slotabl').lower()
open(f'results/slot_ablation_{tag}.json', 'w').write(json.dumps(res, ensure_ascii=False))

VARIANTS = ['control', 'bounded_crit', 'no_crit', 'bounded_plan', 'bounded_both',
            'shuffled_plan', 'foreign_plan', 'no_plan']
ok = [r for r in res if 'error' not in r and len(r['variants']) == len(VARIANTS)]
print(f'\nusable: {len(ok)}/{len(res)}')
ctrl = 'control'
print(f"\n{'variant':16}{'answer unchanged':>18}{'correct':>10}{'c->w':>7}{'w->c':>7}{'prompt chars':>14}")
for v in VARIANTS:
    same = sum(1 for r in ok if r['variants'][v]['norm'] == r['variants'][ctrl]['norm'])
    corr = sum(1 for r in ok if r['variants'][v]['correct'])
    cw = sum(1 for r in ok if r['variants'][ctrl]['correct'] and not r['variants'][v]['correct'])
    wc = sum(1 for r in ok if not r['variants'][ctrl]['correct'] and r['variants'][v]['correct'])
    pc = sum(r['variants'][v]['prompt_chars'] for r in ok) / max(len(ok), 1)
    print(f'{v:16}{same/len(ok)*100:>17.0f}%{corr/len(ok)*100:>9.0f}%{cw:>7}{wc:>7}{pc:>14,.0f}')
