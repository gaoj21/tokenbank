"""Offline half-frontier of the governed arm.

Loosening the gate thresholds (theta_l - delta, delta >= 0) only makes tasks stop earlier, and
every task that escalated past a rung carries that rung's answer, correctness and
log-probabilities. So the looser half of the frontier can be replayed exactly in outcome.
Cost of an earlier stop is estimated from the tasks that actually stopped at that rung:
ratio = total_used / (sum of recorded executor prompt+completion up to that rung), fitted per
cell and per whether a tool was used, then applied to the replayed tasks. delta = 0 must
reproduce the observed rung and accuracy exactly; the script asserts it.
"""
import json, os, sys, collections, statistics as st
sys.path.insert(0, 'analysis'); from _files import path, GOV
TH0, TH1, TAU = -0.10, -0.15, 0.6
DELTAS = [0, 0.05, 0.10, 0.20, 0.40, 9]

def conf(r, th):
    lp, ent = r.get('mean_logprob'), r.get('entropy')
    return lp is not None and lp > th and (ent is None or ent <= TAU)

def load(fam, cfg, t, k):
    p = path(fam, cfg, t, k)
    if not p or not os.path.exists(p): return []
    out = []
    for l in open(p):
        try: d = json.loads(l)
        except: continue
        if d['config'] == cfg: out.append(d)
    return out

def recorded(d, upto):
    rs = [r for r in d['rounds'] if r['role'] == 'executor'][:upto + 1]
    return sum(r['prompt_tokens'] + r['completion_tokens'] for r in rs), any(r.get('tool_output_tokens') for r in rs)

def replay(items, delta):
    # cost ratios from tasks that really stopped at each rung
    ratio = {}
    for lvl in (0, 1):
        for tool in (False, True):
            xs = [d['total_used'] / max(recorded(d, lvl)[0], 1) for d in items
                  if d['escalation_level'] == lvl and recorded(d, lvl)[1] == tool]
            ratio[(lvl, tool)] = st.median(xs) if xs else None
    for lvl in (0, 1):
        for tool in (False, True):
            if ratio[(lvl, tool)] is None:
                ratio[(lvl, tool)] = ratio[(lvl, not tool)] or 1.0
    acc, tok, lv = [], [], collections.Counter()
    for d in items:
        rs = [r for r in d['rounds'] if r['role'] == 'executor']
        stop = None
        if rs and conf(rs[0], TH0 - delta): stop = 0
        elif d['escalation_level'] >= 1 and len(rs) > 1 and conf(rs[1], TH1 - delta): stop = 1
        if stop is None or stop >= d['escalation_level']:
            acc.append(bool(d['success'])); tok.append(d['total_used']); lv[d['escalation_level']] += 1
        else:
            c, tool = recorded(d, stop)
            acc.append(bool(rs[stop].get('is_correct'))); tok.append(c * ratio[(stop, tool)]); lv[stop] += 1
    return 100 * st.mean(acc), st.mean(tok), lv

res = {}
print(f"{'cell':<20}" + ''.join(f'  δ={x:<4}' if x < 9 else '  all-L0 ' for x in DELTAS) + '   | Naive+Fmt 8k / 16k / 24k')
for fam in ['Qwen', 'Gemma', 'GLM', 'DeepSeek']:
    for t, ds in [('ma', 'MATH'), ('ga', 'GAIA'), ('fr', 'FRAMES'), ('mu', 'MuSiQue')]:
        base = {}
        for k in (8, 16, 24):
            b = load(fam, 'V5-Unc-Fmt', t, k)
            base[k] = (100 * st.mean(bool(d['success']) for d in b), st.mean(d['total_used'] for d in b)) if b else None
        for k in (24,):
            items = load(fam, GOV[fam], t, k)
            if not items: continue
            a0, t0, _ = replay(items, 0)
            real = (100 * st.mean(bool(d['success']) for d in items), st.mean(d['total_used'] for d in items))
            assert abs(a0 - real[0]) < 1e-9 and abs(t0 - real[1]) < 1e-6, (fam, t, a0, real)
            pts = [replay(items, x)[:2] for x in DELTAS]
            res[f'{fam}|{ds}|{k}'] = dict(front=pts, base=base)
            print(f"{fam+' '+ds:<20}" + ''.join(f'{a:5.1f}/{tk/1000:4.1f}k' for a, tk in pts)
                  + '   | ' + '  '.join(f'{v[0]:.1f}/{v[1]/1000:.1f}k' if v else '-' for v in base.values()))
json.dump(res, open('results/pareto_half.json', 'w'), indent=1)
print('\nδ=0 replay reproduces observed accuracy and tokens exactly in every cell (asserted).')
