"""N1 step 5: stratified sample for a two-annotator audit of FRAMES and MuSiQue labels.
For each model, CLIMB is calibrated on the full pool at eps = 10% (the tolerance at which it
stops early often enough to disagree with the reference), and tasks are stratified by
{reference judged correct, wrong} x {CLIMB's answer judged correct, wrong}. Per model and benchmark:
25 tasks from each disagreeing cell and 12-13 from each agreeing cell (all if fewer), 75 per
benchmark, 300 in total. Writes paper/data/annotation/sample.csv (annotators fill A_ref, A_pi,
B_ref, B_pi with 1/0) and key.csv (the judge's labels, kept apart for blind annotation)."""
import csv, json, re, sys, numpy as np, pandas as pd
sys.path.insert(0, 'paper/analysis'); sys.path.insert(0, 'analysis')
from climb_core import load, outcome_matrix, calibrate, GRID, TAU
from risk_gate import CELLS, _climbx_path
from datasets import load_dataset
ANS = lambda s: (re.findall(r'ANSWER:\s*(.+)', s or '') or [''])[-1].strip()
EX = lambda d: [r for r in d['rounds'] if r['role'] == 'executor']
Q = {}
for i, it in enumerate(load_dataset('google/frames-benchmark', split='test')):
    Q[f"frames_{it.get('Unnamed: 0', i)}"] = (it['Prompt'], it['Answer'])
for it in load_dataset('dgslibisey/MuSiQue', split='validation'):
    Q['musique_' + it['id']] = (it['question'], ' | '.join([it['answer']] + list(it.get('answer_aliases') or [])))
d = load(); rng = np.random.default_rng(7); rows, keys = [], []
for fam, cells in CELLS.items():
    for ds, ae, nf in cells:
        if ds not in ('FRAMES', 'MuSiQue'): continue
        x = d[(d.model == fam) & (d.bench == ds)].reset_index(drop=True)
        Y, C, L = outcome_matrix(x); j = calibrate(L, C, 0.10); t = GRID[j - 1] if j else 1e9
        s0 = (x.S0 > t) & (x.H0 <= TAU); s1 = ~s0 & (x.S1 > t) & (x.H1 <= TAU)
        rung = np.where(s0, 0, np.where(s1, 1, 2)); ypi = Y[j]
        cx = {json.loads(l)['task_id']: json.loads(l) for l in open(_climbx_path(ae))}
        rf = {json.loads(l)['task_id']: json.loads(l) for l in open(nf) if json.loads(l)['config'] == 'V5-Unc-Fmt'}
        for (yk, yp), k in (((1, 0), 25), ((0, 1), 25), ((1, 1), 13), ((0, 0), 12)):
            idx = np.nonzero((x.YK.to_numpy() == yk) & (ypi == yp))[0]
            for i in rng.permutation(idx)[:k]:
                tid = x.task_id[i]; ex = EX(cx[tid]); r = rung[i]
                a_pi = ANS(ex[r].get('answer_snapshot') if r < 2 else ex[-1].get('answer_snapshot'))
                a_ref = ANS(EX(rf[tid])[-1].get('answer_snapshot'))
                uid = f'{fam[:2]}-{ds[:2]}-{len(rows):03d}'
                rows.append(dict(uid=uid, question=Q[tid][0], gold=Q[tid][1], answer_ref=a_ref, answer_pi=a_pi,
                                 A_ref='', A_pi='', B_ref='', B_pi=''))
                keys.append(dict(uid=uid, model=fam, bench=ds, task_id=tid, threshold=t, rung=int(r), judge_ref=int(yk), judge_pi=int(yp),
                                 cut_ref=int(x.UK[i]), cut_pi=int(x[f'U{r}'][i] if r < 2 else x.U2[i])))
order = rng.permutation(len(rows))                                       # shuffle so strata are not visible
pd.DataFrame([rows[i] for i in order]).to_csv('paper/data/annotation/sample.csv', index=False)
pd.DataFrame([keys[i] for i in order]).to_csv('paper/data/annotation/key.csv', index=False)
print(len(rows), pd.DataFrame(keys).groupby(['model', 'bench', 'judge_ref', 'judge_pi']).size())
