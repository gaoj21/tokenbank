"""Direction A, part 2: does the per-round cost model predict who runs out of budget?

Fit each regime's per-round cost (prompt + completion of the executor round, plus the mean
reviewer cost per round) and derive the affordable round count R_B:
  accumulate:  c_t = c0 + a t   ->  R c0 + a R^2/2 = B  ->  R_B ~ sqrt(2B/a) for large B
  rebuild:     c_t = c          ->  R_B = B / c
Then compare against the observed share of items stopped by the budget.
"""
import json, os, collections, statistics as st, math
import numpy as np
ARM = {'Accumulate':'V5-Accum-Only','Naive+Fmt':'V5-Unc-Fmt','Governed':'V5.1-Ab7-NoVerbal'}
PAT = {'Accumulate':'traces/tasks_sweep_Qwen3.8-27B_acc_{t}{k}k.jsonl',
       'Naive+Fmt':'traces/tasks_sweep_Qwen3.8-27B_q38_{t}{k}.jsonl',
       'Governed':'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{t}{k}.jsonl'}
def rows(arm, k):
    for t in ['ma','ga','fr','mu']:
        p = PAT[arm].format(t=t, k=k)
        if not os.path.exists(p): continue
        for l in open(p):
            try: d = json.loads(l)
            except: continue
            if d['config'] == ARM[arm]: yield d
print(f"{'arm':<12}{'B':>4}{'c0':>7}{'slope':>7}{'R_B pred':>10}{'mean rounds':>13}{'budget-stopped':>16}")
for arm in ARM:
    for k in ['8','16','24']:
        per = collections.defaultdict(list); rv=[]; stop=collections.Counter(); nr=[]
        for d in rows(arm, k):
            rs=[r for r in d['rounds'] if r['role']=='executor']
            if arm=='Governed':
                if d.get('escalation_level',0)<2: continue
                rs=rs[2:]
            for i,r in enumerate(rs[:5]): per[i].append(r['prompt_tokens']+r['completion_tokens'])
            rv.append(d['role_tokens'].get('reviewer',0)/max(len(rs),1))
            stop[d['stopped_by']]+=1; nr.append(len(rs))
        m=[st.mean(per[i]) for i in range(5) if len(per[i])>=20]
        a,c0=np.polyfit(np.arange(len(m)),m,1)
        c0+=st.mean(rv)
        B=int(k)*1000
        if arm=='Accumulate' and a>50:
            R=(-c0+math.sqrt(c0*c0+2*a*B))/a
        else:
            R=B/(st.mean(m[1:])+st.mean(rv))
        tot=sum(stop.values())
        print(f"{arm:<12}{k+'k':>4}{c0:>7.0f}{a:>+7.0f}{R:>10.1f}{st.mean(nr):>13.2f}{100*stop.get('budget',0)/tot:>15.1f}%")
