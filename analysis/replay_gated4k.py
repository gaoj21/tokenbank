"""Replay the compressed pipeline's confidence gate on the full-counterfactual traces at m=4096.
Every item in those traces ran L0, L1 and L2 and recorded each level's answer, correctness and
log-probabilities, so the gated system's accuracy is exact. Its cost is the executor tokens
recorded up to the stopping level (L0 or L1) or the item's total (L2); tool-output notes and
the reviewer are not in the L0/L1 records, so early-stop costs are slight underestimates."""
import json, os, statistics as st
TH0, TH1, TAU = -0.10, -0.15, 0.6
def conf(r, th):
    lp, e = r.get('mean_logprob'), r.get('entropy')
    return lp is not None and lp > th and (e is None or e <= TAU)
R = lambda d: [r for r in d['rounds'] if r['role'] == 'executor']
CELLS = [('GLM','MATH','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_ma24.jsonl'),
         ('GLM','GAIA','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_ga24.jsonl'),
         ('GLM','FRAMES','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_fr24.jsonl'),
         ('GLM','MuSiQue','traces/tasks_sweep_GLM-5.3-Flash_glmx4ka_mu24.jsonl'),
         ('DeepSeek','MATH','traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_ma24.jsonl'),
         ('DeepSeek','GAIA','traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_ga24.jsonl'),
         ('DeepSeek','FRAMES','traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_fr24.jsonl'),
         ('DeepSeek','MuSiQue','traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_mu24.jsonl')]
out = []
for fam, ds, p in CELLS:
    if not os.path.exists(p): continue
    items = [json.loads(l) for l in open(p)]
    items = [d for d in items if d['config'].endswith('AllEsc') and len(R(d)) >= 3]
    if len(items) < 50: print(fam, ds, 'incomplete', len(items)); continue
    acc = []; tok = []; lv = [0, 0, 0]
    for d in items:
        rs = R(d)
        c0 = rs[0]['prompt_tokens'] + rs[0]['completion_tokens']
        c1 = c0 + rs[1]['prompt_tokens'] + rs[1]['completion_tokens']
        if conf(rs[0], TH0): acc.append(bool(rs[0]['is_correct'])); tok.append(c0); lv[0] += 1
        elif conf(rs[1], TH1): acc.append(bool(rs[1]['is_correct'])); tok.append(c1); lv[1] += 1
        else: acc.append(bool(d['success'])); tok.append(d['total_used']); lv[2] += 1
    r = dict(fam=fam, ds=ds, n=len(items), acc=100*st.mean(acc), tok=st.mean(tok),
             l0=100*lv[0]/len(items), l2=100*lv[2]/len(items),
             full=100*st.mean(bool(d['success']) for d in items), fulltok=st.mean(d['total_used'] for d in items))
    out.append(r)
    print(f"{fam:<9}{ds:<8} n={r['n']:<5} gated {r['acc']:5.1f}% / {r['tok']:6.0f} tok   stop@L0 {r['l0']:4.1f}%  reach L2 {r['l2']:4.1f}%   | always-L2 {r['full']:5.1f}% / {r['fulltok']:6.0f}")
json.dump(out, open('results/replay_gated4k.json', 'w'), indent=1)
