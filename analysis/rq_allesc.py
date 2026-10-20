"""Per-rung counterfactual from the all-escalate run: every item has L0, L1 and L2 outcomes.
Rungs are read positionally from the executor rounds: [0]=L0, [1]=L1, last=L2 final.
Cost of stopping at L0/L1 is the executor tokens recorded up to that rung; stopping at L2 is
the item's total. Oracle = stop at the cheapest rung that is correct (or L0 if none is)."""
import json, sys, statistics as st, os
from sklearn.metrics import roc_auc_score
TAG = 'DeepSeek-V4.1-Flash'
rows = []
print(f"{'':<8}{'n':>5}{'L0':>7}{'L1':>7}{'L2':>7}{'救回':>7}{'改坏':>6}{'都错':>7}{'oracle':>8}{'tok L0':>8}{'tok L2':>8}{'tok orc':>8}{'AUC(L0对)':>10}{'AUC(救回)':>10}")
for t, ds in [('ma','MATH'),('ga','GAIA'),('fr','FRAMES'),('mu','MuSiQue')]:
    p = f'traces/tasks_sweep_{TAG}_dsx_allesc_{t}24.jsonl'
    if not os.path.exists(p): continue
    L0=[];L1=[];L2=[];c0=[];c1=[];c2=[];lp=[]
    for l in open(p):
        d = json.loads(l)
        rs = [r for r in d['rounds'] if r['role']=='executor']
        if len(rs) < 3: continue
        L0.append(bool(rs[0]['is_correct'])); L1.append(bool(rs[1]['is_correct'])); L2.append(bool(d['success']))
        c0.append(rs[0]['prompt_tokens']+rs[0]['completion_tokens'])
        c1.append(c0[-1]+rs[1]['prompt_tokens']+rs[1]['completion_tokens'])
        c2.append(d['total_used']); lp.append(rs[0].get('mean_logprob') or -9)
    n=len(L0)
    resc=[(not a) and b for a,b in zip(L0,L2)]; harm=[a and not b for a,b in zip(L0,L2)]
    both=[(not a) and (not b) and (not m) for a,m,b in zip(L0,L1,L2)]
    orc=[a or m or b for a,m,b in zip(L0,L1,L2)]
    oc=[c0[i] if L0[i] else (c1[i] if L1[i] else (c2[i] if L2[i] else c0[i])) for i in range(n)]
    a1=roc_auc_score(L0,lp); a2=roc_auc_score(resc,[-x for x in lp]) if any(resc) else float('nan')
    rows.append(dict(ds=ds,n=n,L0=st.mean(L0),L1=st.mean(L1),L2=st.mean(L2),rescue=st.mean(resc),harm=st.mean(harm),
                     both=st.mean(both),oracle=st.mean(orc),c0=st.mean(c0),c2=st.mean(c2),corc=st.mean(oc),auc_c=a1,auc_r=a2))
    r=rows[-1]
    print(f"{ds:<8}{n:>5}{100*r['L0']:>6.1f}%{100*r['L1']:>6.1f}%{100*r['L2']:>6.1f}%{100*r['rescue']:>6.1f}%{100*r['harm']:>5.1f}%{100*r['both']:>6.1f}%{100*r['oracle']:>7.1f}%{r['c0']:>8.0f}{r['c2']:>8.0f}{r['corc']:>8.0f}{a1:>10.2f}{a2:>10.2f}")
json.dump(rows, open('results/rq_allesc_ds.json','w'), indent=1)
