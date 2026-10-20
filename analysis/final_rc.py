"""Section 5: runtime per-call cap vs fixed caps and a ZEBRA-style static split, at equal budget."""
import json, os, statistics as st
from scipy.stats import binomtest
def load(p, cfg):
    o = {}
    if not os.path.exists(p): return o
    for l in open(p):
        d = json.loads(l)
        if d['config'] == cfg: o[d['task_id']] = d
    return o
M = {'GLM': ('GLM-5.3-Flash', 'glm4', {'24': 'glmx4k', '8': 'glmr4k'}, 'glmrc', 'glmss'),
     'DeepSeek': ('DeepSeek-V4.1-Flash', 'ds', {'24': 'dsx4k', '8': 'dsr4k'}, 'dsrc', 'dsrc')}
pool = {}
out = []
for fam, (T, f1, f4, rc, ss) in M.items():
    for t, ds in [('ma', 'MATH'), ('ga', 'GAIA')]:
        for k in ['24', '8']:
            P = lambda tag: f'traces/tasks_sweep_{T}_{tag}_{t}{k}.jsonl'
            A = {'fixed 1024': load(P(f1), 'V5-Unc-Fmt'), 'fixed 4096': load(P(f4[k]), 'V5-Unc-Fmt'),
                 'static split': load(P(ss), 'SS-Static'), 'static split + keep': load(P(ss), 'SS-Static-Keep'),
                 'runtime cap': load(P(rc), 'RC-CapOnly'), 'runtime cap + keep': load(P(rc), 'RC-Adaptive')}
            ids = set.intersection(*[set(v) for v in A.values()])
            acc = {a: 100*st.mean(bool(v[i]['success']) for i in ids) for a, v in A.items()}
            best = max(['fixed 1024', 'fixed 4096'], key=lambda a: acc[a])
            print(f"\n== {fam} {ds} {k}k  n={len(ids)}  best fixed = {best}")
            for a, v in A.items():
                tok = st.mean(v[i]['total_used'] for i in ids); bs = 100*st.mean(v[i]['stopped_by'] == 'budget' for i in ids)
                B = A[best]; b = sum(1 for i in ids if v[i]['success'] and not B[i]['success']); c = sum(1 for i in ids if B[i]['success'] and not v[i]['success'])
                pool.setdefault(a, [0, 0]); pool[a][0] += b; pool[a][1] += c
                p = binomtest(b, b+c, .5).pvalue if b+c else 1
                out.append(dict(fam=fam, ds=ds, k=k, arm=a, acc=acc[a], tok=tok, budget_stop=bs, d=100*(b-c)/len(ids), p=p))
                print(f"  {a:<22}{acc[a]:6.1f}%  tok {tok:6.0f}  budget-stopped {bs:4.0f}%  vs best fixed {100*(b-c)/len(ids):+5.1f} p={p:.2f}")
print("\n== pooled over 8 cells, against the best fixed cap of each cell")
for a, (b, c) in pool.items():
    if a.startswith('fixed'): continue
    print(f"  {a:<22} wins {b:<4} losses {c:<4} p={binomtest(b, b+c, .5).pvalue:.4f}")
json.dump(out, open('results/final_rc.json', 'w'), indent=1)
