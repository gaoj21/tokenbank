"""Cross-family view of the analysis experiments, one row per (family, benchmark) at 24k.
Prints accuracy and mean tokens of: L0 (one cheap-rung call, from the all-escalate trace),
single call at 2k, the governed arm as run, Naive+Fmt at 1024 and 4096, and the all-escalate
L2; plus rescue/harm of L2 over L0 and paired tests against L0."""
import json, os, statistics as st
from scipy.stats import binomtest
FAM = {
 'Qwen':     dict(tag='Qwen3.8-27B', main=lambda t:f'traces/tasks_sweep_Qwen3.8-27B_q38_{t}24.jsonl',
                  gov=lambda t:(f'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{t}24.jsonl' if os.path.exists(f'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{t}24.jsonl') else f'traces/tasks_sweep_Qwen3.8-27B_q38_ab7_{t}.jsonl'),
                  govcfg='V5.1-Ab7-NoVerbal', single=lambda t:f'traces/tasks_sweep_Qwen3.8-27B_qx_single_{t}24.jsonl',
                  n4=lambda t:(f'traces/tasks_sweep_Qwen3.8-27B_qx4k_{t}24.jsonl','V5-Unc-Fmt'),
                  ae=lambda t:(f'traces/tasks_sweep_Qwen3.8-27B_qx_allesc_{t}24.jsonl','V5.1-Ab7-AllEsc')),
 'GLM':      dict(tag='GLM-5.3-Flash', main=lambda t:f'traces/tasks_sweep_GLM-5.3-Flash_glm4_{t}24.jsonl',
                  gov=lambda t:f'traces/tasks_sweep_GLM-5.3-Flash_glm4_{t}24.jsonl', govcfg='V5.2-ReasonLadder',
                  single=lambda t:f'traces/tasks_sweep_GLM-5.3-Flash_glmx_single_{t}24.jsonl',
                  n4=lambda t:((f'traces/tasks_sweep_GLM-5.3-Flash_glmx4kn_{t}24.jsonl' if t=='mu' else f'traces/tasks_sweep_GLM-5.3-Flash_glmx4k_{t}24.jsonl'),'V5-Unc-Fmt'),
                  ae=lambda t:((f'traces/tasks_sweep_GLM-5.3-Flash_glmx4ka_{t}24.jsonl' if t=='mu' else f'traces/tasks_sweep_GLM-5.3-Flash_glmx4k_{t}24.jsonl'),'V5.2-ReasonLadder-AllEsc')),
 'DeepSeek': dict(tag='DeepSeek-V4.1-Flash', main=lambda t:f'traces/tasks_sweep_DeepSeek-V4.1-Flash_ds_{t}24.jsonl',
                  gov=lambda t:f'traces/tasks_sweep_DeepSeek-V4.1-Flash_ds_{t}24.jsonl', govcfg='V5.2-ReasonLadder',
                  single=lambda t:f'traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx_single_{t}24.jsonl',
                  n4=lambda t:(f'traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_{t}24.jsonl','V5-Unc-Fmt'),
                  ae=lambda t:(f'traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_{t}24.jsonl','V5.2-ReasonLadder-AllEsc')),
}
def load(p, cfg):
    o = {}
    if not p or not os.path.exists(p): return o
    for l in open(p):
        try: d = json.loads(l)
        except: continue
        if d['config'] == cfg: o[d['task_id']] = d
    return o
R = lambda d: [r for r in d['rounds'] if r['role'] == 'executor']
def mc(A, B, ids):
    b = sum(1 for i in ids if B[i] and not A[i]); c = sum(1 for i in ids if A[i] and not B[i])
    p = binomtest(b, b + c, .5).pvalue if b + c else 1
    return 100 * (b - c) / len(ids), p
out = []
hdr = f"{'':<9}{'':<8}{'n':>5} | {'L0':>11} {'单次2k':>11} {'Governed':>11} {'NF-1024':>11} {'NF-4096':>11} {'L2(全升档)':>12} | {'救回':>5} {'改坏':>5} | {'L0→NF4096':>14} {'L0→L2':>14}"
print(hdr)
for fam, F in FAM.items():
    for t, ds in [('ma','MATH'),('ga','GAIA'),('fr','FRAMES'),('mu','MuSiQue')]:
        AE = load(*F['ae'](t)); N4 = load(*F['n4'](t))
        if not AE or not N4: continue
        S2 = load(F['single'](t), 'V5-Single-Fmt-2k'); G = load(F['gov'](t), F['govcfg']); N1 = load(F['main'](t), 'V5-Unc-Fmt')
        ids = [i for i in set(AE) & set(N4) & set(S2) & set(G) & set(N1) if len(R(AE[i])) >= 1]
        if len(ids) < 50: continue
        ok = lambda X: {i: bool(X[i]['success']) for i in ids}
        L0 = {i: bool(R(AE[i])[0]['is_correct']) for i in ids}
        l0tok = st.mean(R(AE[i])[0]['prompt_tokens'] + R(AE[i])[0]['completion_tokens'] for i in ids)
        cell = lambda X: f"{100*st.mean(ok(X).values()):4.1f}/{st.mean(X[i]['total_used'] for i in ids)/1000:4.1f}k"
        res = 100 * st.mean((not L0[i]) and AE[i]['success'] for i in ids)
        harm = 100 * st.mean(L0[i] and not AE[i]['success'] for i in ids)
        d1, p1 = mc(L0, ok(N4), ids); d2, p2 = mc(L0, ok(AE), ids)
        print(f"{fam:<9}{ds:<8}{len(ids):>5} | {100*st.mean(L0.values()):4.1f}/{l0tok/1000:4.1f}k {cell(S2):>11} {cell(G):>11} {cell(N1):>11} {cell(N4):>11} {cell(AE):>12} | {res:5.1f} {harm:5.1f} | {d1:+5.1f} p={p1:.0e} {d2:+5.1f} p={p2:.0e}")
        out.append(dict(fam=fam, ds=ds, n=len(ids), L0=100*st.mean(L0.values()), rescue=res, harm=harm, d_nf4=d1, p_nf4=p1, d_l2=d2, p_l2=p2))
json.dump(out, open('results/rq_crossfam.json', 'w'), indent=1)
