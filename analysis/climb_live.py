"""Live CLIMB on the test halves vs the plain pipeline on the same items, and vs the offline
replay prediction. Thresholds were certified on the disjoint calibration halves
(results/climb_live_plan.json). Paired exact McNemar on accuracy."""
import json, statistics as st, sys, collections
from scipy.stats import binomtest
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items, apply
P = json.load(open('results/climb_live_plan.json'))
TR = {'GLM': 'GLM-5.3-Flash', 'DeepSeek': 'DeepSeek-V4.1-Flash'}
AB = {'MATH': 'ma', 'GAIA': 'ga', 'FRAMES': 'fr', 'MuSiQue': 'mu'}
out = []
print(f"{'cell':<18}{'n':>5}  {'pipeline':>14}  {'offline CLIMB':>18}  {'live CLIMB':>18}  {'Δacc':>6} {'p':>6}  stop L0/L1/L2")
for fam, cells in CELLS.items():
    for ds, ae, nf in cells:
        p = P[f'{fam}|{ds}']; off = {x['id']: x for x in items(ae, nf, keep_id=True)}
        te = [off[i] for i in p['test_ids']]
        pa, pc = 100 * st.mean(x['okP'] for x in te), st.mean(x['cP'] for x in te)
        if p.get('threshold_v3', p.get('threshold_v2', p['threshold'])) == 'direct':
            print(f"{fam+' '+ds:<18}{len(te):>5}  {pa:6.1f}%/{pc/1000:5.1f}k  certified 'direct': runs the pipeline"); 
            out.append(dict(fam=fam, ds=ds, n=len(te), direct=True, pipe_acc=pa, pipe_tok=pc)); continue
        r = [apply(x, p.get('threshold_v3', p.get('threshold_v2', p['threshold']))) for x in te]
        oa, oc = 100 * st.mean(o for o, c in r), st.mean(c for o, c in r)
        TH = p.get('threshold_v3', p.get('threshold_v3', p.get('threshold_v2', p['threshold'])))
        tag = 'climb3'
        path = f'traces/tasks_sweep_{TR[fam]}_{tag}_{AB[ds]}24.jsonl'
        import os
        if not os.path.exists(path):
            print(f"{fam+' '+ds:<18} threshold {TH}: live run pending"); continue
        L = {json.loads(l)['task_id']: json.loads(l) for l in open(path)}
        live = [L[i] for i in p['test_ids']]
        la, lc = 100 * st.mean(d['success'] for d in live), st.mean(d['total_used'] for d in live)
        b = sum(1 for x, d in zip(te, live) if x['okP'] and not d['success']); c = sum(1 for x, d in zip(te, live) if not x['okP'] and d['success'])
        pv = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        lv = collections.Counter(d['escalation_level'] for d in live)
        print(f"{fam+' '+ds:<18}{len(te):>5}  {pa:6.1f}%/{pc/1000:5.1f}k  {oa:6.1f}%/{100*oc/pc:4.0f}% tok  {la:6.1f}%/{100*lc/pc:4.0f}% tok  {la-pa:+6.1f} {pv:6.2f}  {lv[0]}/{lv[1]}/{lv[2]}")
        out.append(dict(fam=fam, ds=ds, n=len(te), theta=p.get('threshold_v3', p.get('threshold_v2', p['threshold'])), pipe_acc=pa, pipe_tok=pc, off_acc=oa, off_tok=oc, live_acc=la, live_tok=lc,
                        harm=b, rescue=c, p=pv, levels=dict(lv)))
json.dump(out, open('results/climb_live.json', 'w'), indent=1)
