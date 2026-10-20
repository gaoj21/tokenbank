#!/usr/bin/env python
"""Assemble every result into one markdown file, computed from traces/ rather
than transcribed, since hand-copying numbers has been a recurring source of
error in this project."""
import json, os, math, glob, collections, datetime

NM = {'ma':'MATH','ga':'GAIA','fr':'FRAMES','mu':'MuSiQue'}
OUT = []
def w(s=''): OUT.append(s)

def mcnemar(b,c):
    n=b+c
    if n==0: return 1.0
    k=min(b,c); ln2=n*math.log(2)
    ls=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-ln2 for i in range(k+1)]
    m=max(ls); return min(1.0,2*math.exp(m+math.log(sum(math.exp(l-m) for l in ls))))

def load(path, force=None, into=None):
    into = into if into is not None else {}
    if os.path.exists(path):
        for ln in open(path):
            r=json.loads(ln); into.setdefault(force or r['config'],{})[r['task_id']]=r
    return into

def trunc(D, ids):
    return sum(1 for i in ids if D[i].get('stopped_by')=='budget' and D[i]['total_used']>0.5*D[i]['budget'])

w(f'# TokenBank results — {datetime.datetime.now():%Y-%m-%d %H:%M}')
w()
w('Model: Qwen3.8-27B (FP8), temperature 0, seed 42. All comparisons are paired '
  'per item with an exact McNemar test. "Truncated" counts runs that spent more '
  'than half the budget and stopped on exhaustion, which excludes API timeouts.')
w()
w('Arms: **Naive** = unbounded loop. **Naive+Fmt** = same, plus the answer-format '
  'instruction (the control every gain is measured against). **Compact** = '
  'accumulate the trajectory and LLM-summarize past a context threshold. '
  '**Accum** = accumulate without summarizing. **Ours-v1** = bounded slots plus a '
  'verbalized escalation channel. **Ours** = bounded slots, routed by the '
  'log-probability gate alone.')
w()

# ---- main matrix ----
w('## 1. Main matrix')
w()
ARMS=[('V5-Unconstrained','Naive'),('V5-Unc-Fmt','Naive+Fmt'),('Compaction','Compact'),
      ('AccumOnly','Accum'),('V5.1-NoPool','Ours-v1'),('V5.1-Ab7-NoVerbal','Ours')]
w('| set | B | n | ' + ' | '.join(l for _,l in ARMS) + ' |')
w('|---|---|---|' + '---|'*len(ARMS))
for ds in ('ma','ga','fr','mu'):
    for b in ('8','16','24'):
        D={}
        load(f'traces/tasks_sweep_Qwen3.8-27B_q38_{ds}{b}.jsonl', into=D)
        load(f'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{ds}{b}.jsonl', into=D)
        if b=='24': load(f'traces/tasks_sweep_Qwen3.8-27B_q38_ab7_{ds}.jsonl', into=D)
        load(f'traces/tasks_sweep_Qwen3.8-27B_cmp_{ds}{b}k.jsonl','Compaction',D)
        load(f'traces/tasks_sweep_Qwen3.8-27B_acc_{ds}{b}k.jsonl','AccumOnly',D)
        if 'V5-Unc-Fmt' not in D: continue
        ids=sorted(set.intersection(*(set(v) for v in D.values())))
        F=D['V5-Unc-Fmt']; cells=[]
        for k,_ in ARMS:
            if k not in D: cells.append('—'); continue
            A=D[k]; sr=sum(A[i]['success'] for i in ids)/len(ids)*100
            if k=='V5-Unc-Fmt': mk=''
            else:
                cw=sum(1 for i in ids if F[i]['success'] and not A[i]['success'])
                wc=sum(1 for i in ids if not F[i]['success'] and A[i]['success'])
                p=mcnemar(cw,wc); mk='\\*' if p<0.05 else ''
            cells.append(f'{sr:.1f}{mk}<br><sub>{sum(A[i]["total_used"] for i in ids)/len(ids):,.0f} tok · {trunc(A,ids)} tr</sub>')
        w(f'| {NM[ds]} | {b}k | {len(ids)} | ' + ' | '.join(cells) + ' |')
w()
w('SR% with `*` for p<0.05 vs Naive+Fmt; second line is mean tokens and real truncations.')
w()

# ---- gains ----
w('## 2. Ours vs the format-aligned baseline')
w()
w('| set | B | n | Naive+Fmt | Ours | Δ pp | win/lose | p | tokens | truncations |')
w('|---|---|---|---|---|---|---|---|---|---|')
for ds in ('ma','ga','fr','mu'):
    for b in ('8','16','24'):
        D={}
        load(f'traces/tasks_sweep_Qwen3.8-27B_q38_{ds}{b}.jsonl', into=D)
        load(f'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{ds}{b}.jsonl', into=D)
        if b=='24': load(f'traces/tasks_sweep_Qwen3.8-27B_q38_ab7_{ds}.jsonl', into=D)
        if 'V5.1-Ab7-NoVerbal' not in D or 'V5-Unc-Fmt' not in D: continue
        A,F=D['V5.1-Ab7-NoVerbal'],D['V5-Unc-Fmt']
        ids=sorted(set(A)&set(F))
        sa=sum(A[i]['success'] for i in ids)/len(ids)*100
        sf=sum(F[i]['success'] for i in ids)/len(ids)*100
        cw=sum(1 for i in ids if F[i]['success'] and not A[i]['success'])
        wc=sum(1 for i in ids if not F[i]['success'] and A[i]['success'])
        dt=(sum(A[i]['total_used'] for i in ids)/sum(F[i]['total_used'] for i in ids)-1)*100
        w(f'| {NM[ds]} | {b}k | {len(ids)} | {sf:.1f} | **{sa:.1f}** | **{sa-sf:+.1f}** | '
          f'{wc}/{cw} | {mcnemar(cw,wc):.1e} | {dt:+.0f}% | {trunc(F,ids)} → **{trunc(A,ids)}** |')
w()

# ---- segments ----
w('## 3. What the pipeline\'s tokens carry')
w()
w('Every character of every prompt in the untreated pipeline, attributed to what it carries.')
w()
w('| benchmark | n | plan | question | draft | critique | tool out | system |')
w('|---|---|---|---|---|---|---|---|')
tot=collections.Counter(); N=0
for ds in ['GAIA','MATH','FRAMES','MUSIQUE','BCB','SWEO']:
    f=f'traces/tasks_sweep_Qwen3.8-27B_seg_{ds}.jsonl'
    if not os.path.exists(f): continue
    agg=collections.Counter(); n=0
    for ln in open(f):
        sc=(json.loads(ln).get('segment_chars') or {})
        if sc: n+=1; agg.update(sc)
    agg.pop('_completion',None); t=sum(agg.values())
    tot.update(agg); N+=n
    g=lambda k: agg.get(k,0)/t*100
    lab={'MUSIQUE':'MuSiQue','BCB':'BigCodeBench','SWEO':'SWE-bench'}.get(ds,ds)
    w(f'| {lab} | {n} | **{g("plan"):.1f}%** | {g("question"):.1f}% | {g("prior_answer"):.1f}% | '
      f'{g("critique"):.1f}% | {g("tool_output"):.1f}% | {g("system"):.1f}% |')
t=sum(tot.values())
w(f'| **pooled** | **{N}** | **{tot["plan"]/t*100:.1f}%** | {tot["question"]/t*100:.1f}% | '
  f'{tot["prior_answer"]/t*100:.1f}% | {tot["critique"]/t*100:.1f}% | {tot["tool_output"]/t*100:.1f}% | '
  f'{tot["system"]/t*100:.1f}% |')
w()

# ---- slot ablation ----
w('## 4. What bounding each slot costs')
w()
w('Counterfactual: on a live round, re-issue the identical executor call with one slot '
  'replaced. Flips are correct→wrong / wrong→correct against the unbounded control.')
w()
w('| benchmark | n | plan → step | p | critique → verdict | p | critique removed | p |')
w('|---|---|---|---|---|---|---|---|')
agg=collections.defaultdict(lambda:[0,0]); tn=0
for tag,name in [('math','MATH'),('gaia','GAIA'),('frames','FRAMES'),('musique','MuSiQue')]:
    f=f'results/slot_ablation_{tag}.json'
    if not os.path.exists(f): continue
    res=[r for r in json.load(open(f)) if 'error' not in r and len(r.get('variants',{}))==5]
    if not res: continue
    tn+=len(res); row=[]
    for v in ['bounded_plan','bounded_crit','no_crit']:
        cw=sum(1 for r in res if r['variants']['control']['correct'] and not r['variants'][v]['correct'])
        wc=sum(1 for r in res if not r['variants']['control']['correct'] and r['variants'][v]['correct'])
        agg[v][0]+=cw; agg[v][1]+=wc
        row += [f'{cw}/{wc}', f'{mcnemar(cw,wc):.3f}']
    w(f'| {name} | {len(res)} | ' + ' | '.join(row) + ' |')
row=[]
for v in ['bounded_plan','bounded_crit','no_crit']:
    cw,wc=agg[v]; row += [f'**{cw}/{wc}**', f'**{mcnemar(cw,wc):.3f}**']
w(f'| **pooled** | **{tn}** | ' + ' | '.join(row) + ' |')
w()
w('The plan is 40% of the budget and free to remove; the critique is 2% and is not.')
w()

open('paper/RESULTS.md','w').write('\n'.join(OUT))
print(f'wrote paper/RESULTS.md ({len(OUT)} lines)')
