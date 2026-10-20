"""Contribution 2: where a pipeline's budget goes, by role, and within the prompts by slot.
Role shares use role_tokens (every call each role made, prompt + completion). Prompt-slot
shares use segment_chars from the runs with SEGMENT_ACCOUNTING=1 (Naive+Fmt, 16k, cap 4096),
pooled over all calls of all roles."""
import json, os, glob, collections, statistics as st
def rows(p, cfg):
    if not os.path.exists(p): return []
    out = []
    for l in open(p):
        try: d = json.loads(l)
        except: continue
        if d['config'] == cfg: out.append(d)
    return out
print("== role shares of total spend, Naive+Fmt, cap 4096, 24k")
for fam, pat, mu in [('GLM', 'traces/tasks_sweep_GLM-5.3-Flash_glmx4k_{t}24.jsonl', 'traces/tasks_sweep_GLM-5.3-Flash_glmfp8n_mu24.jsonl'),
                     ('DeepSeek', 'traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_{t}24.jsonl', None)]:
    for t, ds in [('ma','MATH'),('ga','GAIA'),('fr','FRAMES'),('mu','MuSiQue'),('ALL',None)]:
        ts = ['ma','ga','fr','mu'] if t == 'ALL' else [t]
        c = collections.Counter(); tot = 0
        for tt in ts:
            for d in rows(mu if (tt == 'mu' and mu) else pat.format(t=tt), 'V5-Unc-Fmt'):
                for r, v in d.get('role_tokens', {}).items(): c[r] += v
                tot += d['total_used']
        if not tot: continue
        s = sum(c.values())
        lab = ds or 'pooled'
        print(f"  {fam:<9}{lab:<8} planner {100*c['planner']/s:5.1f}%  executor {100*c['executor']/s:5.1f}%  reviewer {100*c['reviewer']/s:5.1f}%   (roles sum {100*s/tot:.0f}% of total_used)")
print("\n== prompt characters by slot, Naive+Fmt, cap 4096, 16k, all calls (segment accounting)")
for fam, files in [('GLM', sorted(glob.glob('traces/tasks_sweep_GLM-5.3-Flash_glmseg_*.jsonl'))),
                   ('DeepSeek', sorted(glob.glob('traces/tasks_sweep_DeepSeek-V4.1-Flash_dsr4k_*16.jsonl')))]:
    c = collections.Counter(); n = 0; got = []
    for f in files:
        for d in rows(f, 'V5-Unc-Fmt') + rows(f, 'BCB-Naive'):
            sc = d.get('segment_chars') or {}
            if sc: c.update(sc); n += 1
        got.append(os.path.basename(f).split('_')[-1].replace('.jsonl',''))
    if not n: print(f"  {fam}: no segment data yet"); continue
    comp = c.pop('_completion', 0); s = sum(c.values())
    print(f"  {fam:<9} items={n} files={got}")
    print("   " + "  ".join(f"{k} {100*v/s:.1f}%" for k, v in c.most_common()))
    print(f"   completion chars / (prompt+completion) = {100*comp/(s+comp):.1f}%")
