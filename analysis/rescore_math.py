"""Re-score every MATH trace with the fixed judge (_math_judge.py). Originals are backed up to
_archive/traces_math_oldjudge/. For each executor round and for the task outcome, the answer is
re-judged from the stored text when it is visible; when the stored text was cut at 2,000
characters before the answer, the original label is kept. A label is set correct if either judge
accepts it (the old judge had no false positives in our checks)."""
import json, glob, os, re, shutil, sys
from datasets import load_dataset
sys.path.insert(0, '.')
from _math_judge import judge_math
G = {f"math_{i}": it['answer'] for i, it in enumerate(load_dataset('HuggingFaceH4/MATH-500', split='test'))}
cut = lambda a: len(a or '') == 2000 and not re.search(r'ANSWER:|\\boxed', a or '')   # old 2,000-char snapshots only
os.makedirs('_archive/traces_math_oldjudge', exist_ok=True)
stats = dict(files=0, rounds=0, rounds_flip=0, rounds_regress=0, tasks=0, task_flip=0, task_regress=0, kept_cut=0)
for f in sorted(glob.glob('traces/tasks_sweep_*.jsonl')):
    lines = open(f).read().splitlines()
    recs = []
    touched = False
    for l in lines:
        try: d = json.loads(l)
        except Exception: recs.append(l); continue
        tid = str(d.get('task_id', ''))
        if not tid.startswith('math_') or 'rounds' not in d or tid not in G:
            recs.append(l); continue
        touched = True; g = G[tid]
        ex = [r for r in d['rounds'] if r.get('role') == 'executor']
        for r in ex:
            a = r.get('answer_snapshot') or ''; old = bool(r.get('is_correct'))
            stats['rounds'] += 1
            if cut(a): stats['kept_cut'] += 1; continue
            new = judge_math(a, g)
            stats['rounds_flip'] += (new and not old); stats['rounds_regress'] += (old and not new)
            r['is_correct'] = old or new
        stats['tasks'] += 1
        if ex and not cut(ex[-1].get('answer_snapshot')):
            old = bool(d.get('success')); new = judge_math(ex[-1].get('answer_snapshot') or '', g)
            stats['task_flip'] += (new and not old); stats['task_regress'] += (old and not new)
            d['success'] = old or new
        d['math_rescored'] = True
        recs.append(json.dumps(d))
    if touched:
        stats['files'] += 1
        dst = os.path.join('_archive/traces_math_oldjudge', os.path.basename(f))
        if not os.path.exists(dst): shutil.copy2(f, dst)
        open(f, 'w').write('\n'.join(recs) + '\n')
print(stats)
