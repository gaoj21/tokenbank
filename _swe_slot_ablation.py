#!/usr/bin/env python
"""Does bounding the plan cost anything on code repair?

The counterfactual slot ablation found the plan slot free across 850 paired
items, but all of them are question answering. On SWE-bench Verified the
governed arm resolves significantly fewer instances than the untreated one
(46 against 14 on instances where both submit a patch, p=4e-5), and the obvious
suspect is the plan: on QA it is a decomposition ("find X, then compute Y"),
while on repair it may carry the diagnosis of the bug itself.

Same design as the QA ablation, with the one change the benchmark forces: the
answer is a patch, so correctness cannot be judged inline and both variants are
written out as prediction files for the official harness to execute.
"""
import os, re, json, random, sys
import concurrent.futures as cf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _swe_oracle import parse_gold, build_patch, render_task, parse_reply
from openai import OpenAI
from datasets import load_dataset

CLIENT = OpenAI(api_key=os.getenv('VLLM_API_KEY', 'EMPTY'),
                base_url=os.getenv('OPENAI_BASE_URL'), timeout=180, max_retries=2)
MODEL = os.getenv('VLLM_MODEL')
EB = {'chat_template_kwargs': {'enable_thinking': False}}
N = int(os.getenv('N', '250'))
W = int(os.getenv('W', '32'))


def ask(messages, max_tokens=1536):
    r = CLIENT.chat.completions.create(model=MODEL, messages=messages, temperature=0,
                                       max_tokens=max_tokens, extra_body=EB)
    return r.choices[0].message.content or ''


SYS = ('You are an Executor fixing a bug. Reply with the corrected code for each '
       'region in the requested format and nothing else.')


def one(item):
    q = render_task(item)
    try:
        # Use the pipeline's planner call verbatim. An earlier version sent the
        # rendered task as a bare user message with a repair-specific system
        # prompt; because that task ends with a fenced code block, the model
        # continued the pattern and echoed "### REGION 1 / ```python" instead of
        # planning. The bounded-plan slot then held a code fence, so the ablation
        # measured a broken planner rather than the mechanism.
        plan = ask([{'role': 'system',
                     'content': 'You are a Planner. Break down the question into sub-tasks.'},
                    {'role': 'user',
                     'content': f'Question: {q}\nProvide a step-by-step plan.'}], max_tokens=768)
        step = _ls[1] if len(_ls) > 1 else (_ls[0] if _ls else plan[:200])
        out = {}
        for name, p in (('control', plan), ('bounded_plan', step)):
            out[name] = ask([{'role': 'system', 'content': SYS},
                             {'role': 'user', 'content': f'{q}\n\nPlan: {p}'}])
        return {'instance_id': item['instance_id'], 'plan_chars': len(plan),
                'step_chars': len(step), 'replies': out}
    except Exception as e:
        return {'instance_id': item['instance_id'], 'error': str(e)[:120]}


ds = list(load_dataset('SWE-bench/SWE-bench_Verified', split='test'))
random.Random(42).shuffle(ds)
ds = ds[:N]
with cf.ThreadPoolExecutor(max_workers=W) as ex:
    res = list(ex.map(one, ds))

by_id = {it['instance_id']: it for it in ds}
ok = [r for r in res if 'error' not in r]
print(f'usable: {len(ok)}/{len(res)}')
if ok:
    print(f'mean plan {sum(r["plan_chars"] for r in ok)/len(ok):.0f} chars '
          f'-> step {sum(r["step_chars"] for r in ok)/len(ok):.0f} chars')
for variant in ('control', 'bounded_plan'):
    preds = []
    for r in ok:
        files = parse_gold(by_id[r['instance_id']]['patch'])
        reps = parse_reply(r['replies'][variant], files)
        preds.append({'instance_id': r['instance_id'],
                      'model_name_or_path': f'slot-{variant}',
                      'model_patch': build_patch(files, reps)})
    path = f'swe_preds_slot_{variant}.json'
    json.dump(preds, open(path, 'w'))
    ne = sum(1 for p in preds if p['model_patch'].strip())
    print(f'{variant}: {ne}/{len(preds)} non-empty -> {path}')
json.dump(res, open('results/swe_slot_ablation_raw.json', 'w'))
