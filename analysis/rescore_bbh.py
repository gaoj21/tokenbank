"""Rescore BBH runs in place with _bbh_judge (every executor round's is_correct and the task's success),
from the stored replies. The pre-rescore file is kept under _archive/traces_bbh_prerescore/.
usage: python analysis/rescore_bbh.py traces/tasks_sweep_*_bbh100.jsonl"""
import glob, json, os, random, shutil, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bbh_judge import judge_bbh

Q = {}
for f in sorted(glob.glob('data/bbh/*/test-*.parquet')):
    n = os.path.basename(os.path.dirname(f)); d = pd.read_parquet(f)
    for i in range(len(d)):
        Q[f'bbh_{n}_{i}'] = (d.input.iloc[i], str(d.target.iloc[i]))
os.makedirs('_archive/traces_bbh_prerescore', exist_ok=True)
for p in sys.argv[1:]:
    dst = os.path.join('_archive/traces_bbh_prerescore', os.path.basename(p))
    if not os.path.exists(dst):
        shutil.copy2(p, dst)
    out, flips = [], 0
    for l in open(dst):
        d = json.loads(l); q, g = Q[d['task_id']]
        ex = [r for r in d['rounds'] if r['role'] == 'executor']
        for r in ex:
            v = judge_bbh(r.get('answer_snapshot'), g, q); flips += v != bool(r.get('is_correct')); r['is_correct'] = v
        if ex:
            d['success'] = judge_bbh(ex[-1].get('answer_snapshot'), g, q)
        d['judge'] = 'bbh'
        out.append(json.dumps(d))
    open(p, 'w').write('\n'.join(out) + '\n')
    print(os.path.basename(p), len(out), 'rounds flipped', flips)
