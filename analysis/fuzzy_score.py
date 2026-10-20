"""Judge-free fuzzy scoring of FRAMES / MuSiQue runs, for a quick look before the LLM judge.
An answer counts as correct if, after the experiment's own extraction (_extract_answer: ANSWER line,
then "answer is", then the last line) and normalisation (lower case, no punctuation or articles),
it equals the gold answer, contains it, or has token F1 >= 0.6 with it.
Gold answers come straight from the datasets; the experiment modules are not imported (importing
_budget_sweep_v2 starts a run).
usage: python analysis/fuzzy_score.py MODEL [trace_dir] [climb_tag ref_tag]"""
import json, re, sys
from collections import Counter
from datasets import load_dataset


def norm(s):
    s = str(s).lower()
    s = re.sub(r'[^\w\s\-\.]', ' ', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def extract(text):
    text = text or ''
    for pat in (r'answer\s*(?:is|are|was|were)?\s*[:\-]\s*(.+)', r'answer\s+(?:is|are|was|were)\s+(.+)'):
        m = list(re.finditer(pat, text, re.I))
        if m:
            return m[-1].group(1).strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()


def fuzzy(pred, gold):
    p, g = norm(extract(pred)), norm(gold)
    if not p or not g:
        return False
    if p == g or g in p:
        return True
    pt, gt = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gt)).values())
    return common > 0 and 2 * common / (len(pt) + len(gt)) >= 0.6


def gold_answers():
    gold = {}
    for i, it in enumerate(load_dataset('google/frames-benchmark', split='test')):
        if it.get('Prompt') and it.get('Answer'):
            gold[f"frames_{it.get('Unnamed: 0', i)}"] = str(it['Answer'])
    for it in load_dataset('dgslibisey/MuSiQue', split='validation'):
        gold[f"musique_{it['id']}"] = str(it['answer'])
    return gold


def score(cpath, rpath, gold):
    C = {json.loads(l)['task_id']: json.loads(l) for l in open(cpath)}
    R = {json.loads(l)['task_id']: json.loads(l) for l in open(rpath)}
    ids = [i for i in sorted(set(C) & set(R)) if i in gold]
    y0, y1, yk = [], [], []
    for i in ids:
        ex = [e for e in C[i]['rounds'] if e['role'] == 'executor']
        y0.append(fuzzy(ex[0].get('answer_snapshot'), gold[i]))
        y1.append(fuzzy(ex[1].get('answer_snapshot'), gold[i]))
        yk.append(fuzzy(R[i]['rounds'][-1].get('answer_snapshot') if R[i]['rounds'] else '', gold[i]))
    n = len(ids); p = lambda v: 100 * sum(v) / n
    no_rounds = sum(not R[i]['rounds'] for i in ids)
    resc = sum((not a) and k for a, k in zip(y0, yk)); harm = sum(a and not k for a, k in zip(y0, yk))
    return (f"n={n} L0 {p(y0):5.1f} L1 {p(y1):5.1f} team {p(yk):5.1f} | rescue {100*resc/n:4.1f} harm {100*harm/n:4.1f} "
            f"| L0->L1 fix {sum((not a) and b for a, b in zip(y0, y1))} break {sum(a and not b for a, b in zip(y0, y1))} | team without rounds {no_rounds}")


if __name__ == '__main__':
    m = sys.argv[1]; d = sys.argv[2] if len(sys.argv) > 2 else 'traces/'
    ct, rt = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ('climbx', 'gx4k')
    gold = gold_answers()
    for b in ('fr', 'mu'):
        print(f"{m} {d} {b}: " + score(f'{d}tasks_sweep_{m}_{ct}_{b}100.jsonl', f'{d}tasks_sweep_{m}_{rt}_{b}100.jsonl', gold))
