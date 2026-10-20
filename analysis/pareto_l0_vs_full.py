"""Is escalation buying accuracy? Governed as run (delta=0) against the replayed policy that
stops every task at L0, paired per item, exact McNemar. Same items, same trace."""
import json, os, sys, statistics as st
from scipy.stats import binomtest
sys.path.insert(0, 'analysis'); from _files import path, GOV
out = []
print(f"{'cell':<18}{'B':>4}{'esc%':>6}{'full':>7}{'L0only':>8}{'Δ':>7}{'b/c':>9}{'p':>9}{'tok full':>9}{'tok L0':>8}")
for fam in ['Qwen','Gemma','GLM','DeepSeek']:
    for t, ds in [('ma','MATH'),('ga','GAIA'),('fr','FRAMES'),('mu','MuSiQue')]:
        for k in (24, 16, 8):
            p = path(fam, GOV[fam], t, k)
            if not p or not os.path.exists(p): continue
            items = [json.loads(l) for l in open(p)]
            items = [d for d in items if d['config'] == GOV[fam]]
            full = [bool(d['success']) for d in items]
            l0 = []; tl0 = []
            for d in items:
                rs = [r for r in d['rounds'] if r['role'] == 'executor']
                l0.append(bool(d['success']) if d['escalation_level'] == 0 or not rs else bool(rs[0].get('is_correct')))
            b = sum(1 for f, z in zip(full, l0) if f and not z)
            c = sum(1 for f, z in zip(full, l0) if z and not f)
            pv = binomtest(b, b + c, .5).pvalue if b + c else 1
            esc = 100 * st.mean(d['escalation_level'] > 0 for d in items)
            out.append(dict(fam=fam, ds=ds, k=k, esc=esc, full=100*st.mean(full), l0=100*st.mean(l0), b=b, c=c, p=pv))
            print(f"{fam+' '+ds:<18}{k:>3}k{esc:>5.0f}%{100*st.mean(full):>7.1f}{100*st.mean(l0):>8.1f}{100*(st.mean(full)-st.mean(l0)):>+7.1f}{f'{b}/{c}':>9}{pv:>9.1e}")
json.dump(out, open('results/pareto_l0_vs_full.json', 'w'), indent=1)
pos = sum(1 for r in out if r['p'] < .05 and r['full'] > r['l0']); neg = sum(1 for r in out if r['p'] < .05 and r['full'] < r['l0'])
print(f'\n{len(out)} cells: escalation significantly helps in {pos}, significantly hurts in {neg}, neither in {len(out)-pos-neg}')
