"""Validate a judge model on the FRAMES/MuSiQue answers of given trace files: (a) answers already
correct under exact match should be judged CORRECT; (b) repeatability of the verdicts at the chosen
decoding; (c) agreement between two reasoning efforts. Uses llm_judge's prompt and parsing.
usage: JUDGE_MODEL=... JUDGE_ENDPOINTS=... python analysis/judge_validate.py FILE_TAG ... -> results/judge_validation_<model>.json"""
import json, os, random, sys, concurrent.futures as cf
sys.path.insert(0, 'analysis')
import llm_judge as J
G = J.golds(); random.seed(0)
em_ok, em_bad = [], []
for tag in sys.argv[1:]:
    for l in open(f'traces/tasks_sweep_{tag}.jsonl'):
        d = json.loads(l)
        if d['task_id'] not in G: continue
        for r in [x for x in d['rounds'] if x['role'] == 'executor']:
            s = r.get('answer_snapshot')
            if J.cut(s) or not J.extract(s): continue
            it = dict(tid=d['task_id'], q=G[d['task_id']][0], gold=G[d['task_id']][1], pred=J.extract(s))
            (em_ok if r.get('is_correct') else em_bad).append(it)
A = random.sample(em_ok, min(60, len(em_ok))); B = random.sample(em_bad, min(150, len(em_bad)))
def verdicts(items, extra):
    os.environ['JUDGE_EXTRA_JSON'] = json.dumps(extra)
    with cf.ThreadPoolExecutor(32) as ex:
        return list(ex.map(lambda a: J.judge_one(a[1], J.ENDPOINTS[a[0] % len(J.ENDPOINTS)])[0], enumerate(items)))
low1 = verdicts(A + B, {'reasoning_effort': 'low'}); low2 = verdicts(A + B, {'reasoning_effort': 'low'})
med = verdicts(A + B, {'reasoning_effort': 'medium'})
nA = len(A)
res = dict(model=J.JUDGE_MODEL, n_em_correct=nA, em_correct_judged_correct_low=sum(v is True for v in low1[:nA]),
           em_correct_judged_correct_medium=sum(v is True for v in med[:nA]), n_em_wrong=len(B),
           em_wrong_judged_correct_low=sum(v is True for v in low1[nA:]), em_wrong_judged_correct_medium=sum(v is True for v in med[nA:]),
           repeat_agreement_low=sum(a == b for a, b in zip(low1, low2)) / len(low1),
           low_vs_medium_agreement=sum(a == b for a, b in zip(low1, med)) / len(low1),
           no_verdict=sum(v is None for v in low1 + low2 + med))
json.dump(dict(res, sample_wrong=[dict(B[i], low=low1[nA + i], medium=med[nA + i]) for i in range(len(B))]),
          open(f"results/judge_validation_{J.JUDGE_MODEL.split('/')[-1]}.json", 'w'), indent=1)
print(json.dumps(res, indent=1))
