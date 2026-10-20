"""Apply the fixed closing call to cheap-rung L1 replies that came back empty (a thinking model filled the
16,384-token call cap). The fixed pipeline answers such a reply with one closing call that has thinking off
(_tool_loop); thinking is never carried in the history, so sending that call now, on the same messages, gives
exactly what a rerun with the fixed code would have produced for these tasks and changes nothing elsewhere.
Updates the L1 round's answer_snapshot, mean_logprob, is_correct and reserve_after (cost), and the task's
success; the pre-patch file is kept under _archive/traces_prepatch_l1/.
usage: python analysis/patch_empty_l1.py BENCH trace.jsonl   (Gemma through OpenRouter/Makora)"""
import json, os, shutil, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = open(os.path.expanduser('~/.config/tokenbank/openrouter.env')).read().split('=', 1)[1].strip()
ANS = ' End your reply with a single line: "ANSWER: <final answer>".'
SYS = ('You are an Executor. Answer concisely.' + ANS + ' For ANY computation, put Python code that prints the result '
       'in a ```python block; you will receive the output, then give the final answer.')
CLOSE = 'Do not write any more code. Give the final answer now, ending with the line ANSWER: <answer>.'


def load_questions(bench):
    """Task id -> question text exactly as the sweep built it (loaders read without running a sweep)."""
    import importlib.util, types
    src = open('_budget_sweep_v2.py').read()
    head = src[:src.index("BENCH = os.getenv('BENCHMARK', 'GAIA')")]
    mod = types.ModuleType('loaders'); mod.__dict__['__file__'] = os.path.abspath('_budget_sweep_v2.py')
    os.environ['TOKENBANK_LOADERS_ONLY'] = '1'
    exec(compile(head, '_budget_sweep_v2.py', 'exec'), mod.__dict__)
    if bench == 'MATH':
        tasks = mod.load_math_dataset(n_tasks=500)
    elif bench == 'BBH':
        tasks = mod.load_bbh(37)
    elif bench == 'MUSIQUE':
        tasks = mod.load_musique_dataset(n_tasks=99999)
    else:
        tasks = mod.FRAMES_TASKS
    return {t['id']: (t['question'], t['answer']) for t in tasks}


def close_call(question):
    body = {'model': 'google/gemma-4-26b-a4b-it', 'max_tokens': 16384, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 64,
            'logprobs': True, 'reasoning': {'enabled': False},
            'provider': {'order': ['Makora'], 'allow_fallbacks': False, 'require_parameters': True},
            'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': f'Question: {question}\nFinal answer.'},
                         {'role': 'assistant', 'content': ''}, {'role': 'user', 'content': CLOSE}]}
    for _ in range(30):
        try:
            req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', json.dumps(body).encode(),
                                         {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'})
            r = json.load(urllib.request.urlopen(req, timeout=900))
            c = r['choices'][0]; toks = [t['logprob'] for t in ((c.get('logprobs') or {}).get('content') or [])]
            return c['message'].get('content') or '', (sum(toks) / len(toks) if toks else None), r['usage']
        except Exception:
            import time; time.sleep(10)
    raise RuntimeError('closing call failed')


if __name__ == '__main__':
    bench, path = sys.argv[1], sys.argv[2]
    # Judges are imported directly: importing _run_v2_experiment executes its notebook cells (a full run).
    def judge_task(tid, content, gold):
        if tid.startswith('math'):
            from _math_judge import judge_math
            return judge_math(content, gold), None
        if tid.startswith('bbh_'):
            from _bbh_judge import judge_bbh
            return judge_bbh(content, gold), None
        return False, None          # FRAMES / MuSiQue: left to the LLM judge, which reads the new snapshot
    Q = load_questions(bench)
    os.makedirs('_archive/traces_prepatch_l1', exist_ok=True)
    dst = os.path.join('_archive/traces_prepatch_l1', os.path.basename(path))
    if not os.path.exists(dst):
        shutil.copy2(path, dst)
    import concurrent.futures as cf
    recs = [json.loads(l) for l in open(dst)]
    todo = []
    for k, d in enumerate(recs):
        ex = [r for r in d['rounds'] if r['role'] == 'executor']
        if len(ex) >= 2 and not (ex[1].get('answer_snapshot') or '').strip():
            todo.append(k)
    def fix(k):
        d = recs[k]; ex = [r for r in d['rounds'] if r['role'] == 'executor']
        q, gold = Q[d['task_id']]
        content, lp, u = close_call(q)
        ok, _ = judge_task(d['task_id'], content, gold)
        used = u['prompt_tokens'] + u['completion_tokens']
        ex[1]['answer_snapshot'] = content; ex[1]['mean_logprob'] = lp; ex[1]['is_correct'] = ok
        ex[1]['reserve_after'] = ex[1]['reserve_after'] - used; ex[1]['patched_close'] = True
        d['success'] = ok; d['total_used'] = d['total_used'] + used
    with cf.ThreadPoolExecutor(int(os.environ.get('PATCH_WORKERS', '32'))) as pool:
        list(pool.map(fix, todo))
    out, n = [json.dumps(d) for d in recs], len(todo)
    open(path, 'w').write('\n'.join(out) + '\n')
    print(os.path.basename(path), 'patched', n)
