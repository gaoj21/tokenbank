"""Direction C: can one pre-measurable quantity predict whether the reasoning rung helps?

For every cell where both the prefix-only arm (Ab7) and the reasoning ladder exist, compute
  delta = SR(ladder) - SR(Ab7)
and candidate predictors measured on the Ab7 arm alone, i.e. before the rung is turned on:
  cap   share of items with any executor round hitting the generation cap
  empty share of items whose final answer is empty
  gshare completion tokens / total tokens (how much of the spend is generation)
"""
import json, os, collections, statistics as st
from scipy.stats import spearmanr, binomtest
CELLS = []
for fam, tag in [('GLM','GLM-5.3-Flash'), ('DeepSeek','DeepSeek-V4.1-Flash')]:
    for k in ['24','16','8']:
        if fam == 'GLM':
            a = f'traces/tasks_sweep_{tag}_glmab7_ma{k}.jsonl'; l = f'traces/tasks_sweep_{tag}_glm4_ma{k}.jsonl'
        else:
            a = f'traces/tasks_sweep_{tag}_dsab7_ma{k}.jsonl'; l = f'traces/tasks_sweep_{tag}_ds_ma{k}.jsonl'
        CELLS.append((fam, 'MATH', k+'k', a, 'V5.1-Ab7-NoVerbal', l, 'V5.2-ReasonLadder'))
    for k in ['16k','8k','4k']:
        f = (f'traces/tasks_sweep_{tag}_glmbcb{k}_rejudged.jsonl' if fam=='GLM'
             else f'traces/tasks_sweep_{tag}_ds_bcb{k}_rejudged.jsonl')
        CELLS.append((fam, 'BCB', k, f, 'BCB-Ab7', f, 'BCB-ReasonLadder'))
def load(p, cfg):
    out = {}
    if not os.path.exists(p): return out
    for x in open(p):
        try: d = json.loads(x)
        except: continue
        if d['config'] == cfg: out[d['task_id']] = d
    return out
pts = []
print(f"{'cell':<22}{'n':>5}{'cap':>7}{'empty':>7}{'gshare':>8}{'delta':>8}{'p':>10}")
for fam, ds, k, fa, ca, fl, cl in CELLS:
    A, L = load(fa, ca), load(fl, cl)
    ids = set(A) & set(L)
    if not ids: print(f'{fam} {ds} {k}: missing'); continue
    cap = st.mean(1 if any(r['completion_tokens'] >= 1000 for r in A[i]['rounds']) else 0 for i in ids)
    emp = st.mean(1 if not (A[i].get('raw_answer') or (A[i]['rounds'] and A[i]['rounds'][-1].get('answer_snapshot'))) else 0 for i in ids)
    gs = st.mean(sum(r['completion_tokens'] for r in A[i]['rounds']) / max(A[i]['total_used'], 1) for i in ids)
    sa = {i: bool(A[i]['success']) for i in ids}; sl = {i: bool(L[i]['success']) for i in ids}
    d = 100 * (sum(sl.values()) - sum(sa.values())) / len(ids)
    b = sum(1 for i in ids if sl[i] and not sa[i]); c = sum(1 for i in ids if sa[i] and not sl[i])
    p = binomtest(b, b + c, .5).pvalue if b + c else 1
    pts.append(dict(fam=fam, ds=ds, k=k, n=len(ids), cap=cap, empty=emp, gshare=gs, delta=d, p=p))
    print(f"{fam+' '+ds+' '+k:<22}{len(ids):>5}{100*cap:>6.0f}%{100*emp:>6.0f}%{100*gs:>7.0f}%{d:>+8.1f}{p:>10.1e}")
for key in ['cap', 'empty', 'gshare']:
    r, p = spearmanr([x[key] for x in pts], [x['delta'] for x in pts])
    print(f'Spearman({key}, delta) = {r:+.2f}  p={p:.3f}  n={len(pts)}')
json.dump(pts, open('results/c_rung_predictor.json', 'w'), indent=1)
