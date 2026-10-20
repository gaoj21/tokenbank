"""LLM-as-judge for FRAMES and MuSiQue (closed-book), replacing strict exact match, which marks
verbose but correct answers wrong. Judge: GLM-5.3-Flash (FP8), temperature 0, low reasoning effort.
Answers already correct under exact match are accepted without a call; judgments are cached by
(task id, normalised prediction).

Usage: python analysis/llm_judge.py collect   -> results/judge_queue.jsonl
       python analysis/llm_judge.py run N     -> judges up to N queued items (all if N=0)
       python analysis/llm_judge.py apply     -> rewrites the traces (originals in _archive/traces_em/)"""
import json, os, re, sys, glob, shutil, string, random, threading, concurrent.futures as cf
from datasets import load_dataset
# Paths, files and decoding are overridable so a different judge model never reads or overwrites
# another judge's verdicts: JUDGE_QUEUE, JUDGE_CACHE, JUDGE_FILES (comma-separated trace tags),
# JUDGE_TEMPERATURE, JUDGE_EXTRA_JSON (extra request body, e.g. a reasoning effort).
QUEUE = os.environ.get('JUDGE_QUEUE', 'results/judge_queue.jsonl')
CACHE = os.environ.get('JUDGE_CACHE', 'results/judge_cache.jsonl')
FILES = ['GLM-5.3-Flash_climbx_fr24', 'GLM-5.3-Flash_climbx_mu24', 'DeepSeek-V4.1-Flash_climbx_fr24', 'DeepSeek-V4.1-Flash_climbx_mu24',
         'GLM-5.3-Flash_glmx4k_fr24', 'GLM-5.3-Flash_glmfp8n_mu24', 'DeepSeek-V4.1-Flash_dsx4k_fr24', 'DeepSeek-V4.1-Flash_dsx4k_mu24',
         'GLM-5.3-Flash_glms4k_fr24', 'GLM-5.3-Flash_glmfp8s_mu24', 'DeepSeek-V4.1-Flash_dss4k_fr24', 'DeepSeek-V4.1-Flash_dss4k_mu24',
         'GLM-5.3-Flash_climb3_fr24', 'GLM-5.3-Flash_climb3_mu24', 'DeepSeek-V4.1-Flash_climb3_fr24', 'DeepSeek-V4.1-Flash_climb3_mu24',
         'GLM-5.3-Flash_n2c1_mu24', 'GLM-5.3-Flash_n2p1_mu24', 'GLM-5.3-Flash_n2c2_mu24', 'GLM-5.3-Flash_n2p2_mu24', 'GLM-5.3-Flash_n2c3_mu24', 'GLM-5.3-Flash_n2p3_mu24',
         'GLM-5.3-Flash_n3s16k_fr24', 'GLM-5.3-Flash_n3s16k_mu24']
if os.environ.get('JUDGE_FILES'):
    FILES = os.environ['JUDGE_FILES'].split(',')
CONFIGS = {'CLIMB-Live', 'V5-Unc-Fmt', 'V5-Single-Fmt-4k', 'V5-Single-Fmt-16k'}
import os as _os
ENDPOINTS = _os.environ.get('JUDGE_ENDPOINTS', 'https://llm3.ruoyu-wang.com/v1,https://llm4.ruoyu-wang.com/v1,https://llm5.ruoyu-wang.com/v1').split(',')
JUDGE_MODEL = _os.environ.get('JUDGE_MODEL', 'GLM-5.3-Flash-FP8')

def golds():
    G = {}
    for i, it in enumerate(load_dataset('google/frames-benchmark', split='test')):
        G[f"frames_{it.get('Unnamed: 0', i)}"] = (it['Prompt'], [it['Answer']])
    for it in load_dataset('dgslibisey/MuSiQue', split='validation'):
        if it.get('answerable', True):
            G[f"musique_{it['id']}"] = (it['question'], [it['answer']] + list(it.get('answer_aliases') or []))
    return G
def extract(snap):
    m = re.findall(r'ANSWER:\s*(.+)', snap or '')
    if m: return m[-1].strip()
    lines = [l.strip() for l in (snap or '').strip().splitlines() if l.strip()]
    return lines[-1] if lines else ''
# a snapshot cut by the old 2,000-character limit (newer snapshots keep the tail, see _snap)
cut = lambda s: len(s or '') == 2000 and 'ANSWER:' not in (s or '')
norm = lambda s: ' '.join(''.join(ch for ch in (s or '').lower() if ch not in string.punctuation).split())
key = lambda tid, pred: f'{tid}||{norm(pred)}'

def collect():
    G = golds(); q = {}
    for f in FILES:
        for l in open(f'traces/tasks_sweep_{f}.jsonl'):
            d = json.loads(l)
            if d['config'] not in CONFIGS or d['task_id'] not in G: continue
            ex = [r for r in d['rounds'] if r['role'] == 'executor']
            if not ex: continue
            targets = [(r.get('answer_snapshot'), r.get('is_correct')) for r in ex] + [(ex[-1].get('answer_snapshot'), d.get('success'))]
            for snap, em in targets:
                if cut(snap) or em: continue
                pred = extract(snap)
                if not pred: continue
                q[key(d['task_id'], pred)] = dict(tid=d['task_id'], q=G[d['task_id']][0], gold=G[d['task_id']][1], pred=pred)
    with open(QUEUE, 'w') as fh:
        for v in q.values(): fh.write(json.dumps(v) + '\n')
    print('queued', len(q))

PROMPT = ("Question: {q}\nGround-truth answer: {g}\nPredicted answer: {p}\n\n"
          "Grade the predicted answer. It is CORRECT only if it gives the same answer as the ground truth: the "
          "same entity, or the same number, date or range (wording, units written differently, and extra "
          "explanation are fine). It is INCORRECT if any specific value differs (a different number, a narrower or "
          "wider range, a year instead of a full date, a different currency or quantity), if it answers only part "
          "of the question, hedges between several answers, or gives no answer. Reply with exactly one word: "
          "CORRECT or INCORRECT.")
def judge_one(item, ep):
    from openai import OpenAI
    cli = OpenAI(base_url=ep, api_key='EMPTY', timeout=120)
    g = item['gold'][0] + (f" (also accepted: {'; '.join(item['gold'][1:4])})" if len(item['gold']) > 1 else '')
    for _ in range(4):
        try:
            r = cli.chat.completions.create(model=JUDGE_MODEL, temperature=float(os.environ.get('JUDGE_TEMPERATURE', '0')), max_tokens=4096,
                                            extra_body=json.loads(os.environ.get('JUDGE_EXTRA_JSON', '{}')) or None,
                                            
                                            messages=[{'role': 'system', 'content': 'You are an impartial grader of short factual answers.'},
                                                      {'role': 'user', 'content': PROMPT.format(q=item['q'], g=g, p=item['pred'])}])
            out = (r.choices[0].message.content or '').strip().upper()
            if 'INCORRECT' in out: return False, out[-40:]
            if 'CORRECT' in out: return True, out[-40:]
        except Exception as e:
            out = f'ERR {e}'[:80]
            import time; time.sleep(5 + 10 * random.random())
    return None, out

def run(limit):
    done = set()
    if os.path.exists(CACHE):
        for l in open(CACHE): done.add(json.loads(l)['k'])
    items = [json.loads(l) for l in open(QUEUE)]
    items = [it for it in items if key(it['tid'], it['pred']) not in done]
    if limit: random.seed(0); items = random.sample(items, min(limit, len(items)))
    lock = threading.Lock(); fh = open(CACHE, 'a'); n = [0, 0]
    def work(i_it):
        i, it = i_it
        v, raw = judge_one(it, ENDPOINTS[i % len(ENDPOINTS)])
        with lock:
            fh.write(json.dumps(dict(k=key(it['tid'], it['pred']), v=v, raw=raw)) + '\n'); fh.flush()
            n[0] += 1; n[1] += v is None
            if n[0] % 500 == 0: print('judged', n[0], 'failed', n[1], flush=True)
    with cf.ThreadPoolExecutor(int(os.environ.get('JUDGE_THREADS', '36'))) as ex: list(ex.map(work, enumerate(items)))
    print('done', n)

def apply():
    C = {}
    for l in open(CACHE):
        d = json.loads(l)
        if d['v'] is not None: C[d['k']] = d['v']
    os.makedirs('_archive/traces_em', exist_ok=True); st = dict(changed=0, missing=0)
    for f in FILES:
        p = f'traces/tasks_sweep_{f}.jsonl'; dst = f'_archive/traces_em/tasks_sweep_{f}.jsonl'
        # Keep the pre-judge (exact-match) copy of the CURRENT run. The backup used to be written only
        # once per file name, so a rerun under the same name was judged from the old run's backup and
        # its new records were overwritten. A file not yet judged (no 'judge' mark) refreshes the backup.
        with open(p) as fh:
            first = fh.readline()
        if not os.path.exists(dst) or (first and json.loads(first).get('judge') != 'llm'):
            shutil.copy2(p, dst)
        out = []
        for l in open(dst):
            d = json.loads(l)
            if d['config'] in CONFIGS and 'rounds' in d:
                ex = [r for r in d['rounds'] if r['role'] == 'executor']
                for r in ex:
                    s = r.get('answer_snapshot')
                    if r.get('is_correct') or cut(s) or not extract(s): continue
                    k = key(d['task_id'], extract(s))
                    if k in C: st['changed'] += C[k]; r['is_correct'] = C[k]
                    else: st['missing'] += 1
                s = ex[-1].get('answer_snapshot') if ex else None
                if ex and not d.get('success') and not cut(s) and extract(s):
                    k = key(d['task_id'], extract(s))
                    if k in C: d['success'] = C[k]
                d['judge'] = 'llm'
            out.append(json.dumps(d))
        open(p, 'w').write('\n'.join(out) + '\n')
    print(st)

if __name__ == '__main__':
    {'collect': collect, 'run': lambda: run(int(sys.argv[2])), 'apply': apply}[sys.argv[1]]()
