"""Contribution 3: carried state x value of revision.

Arms (all start at the full loop, cap 4096): ST-Rebuild = V5-Unc-Fmt, ST-Accum, ST-Window2,
ST-Summary, ST-Slots share the stop rule (stop when the reviewer accepts, or at round 5);
ST-Slots-M1 and ST-Rebuild-M1 use the strict log-probability stop instead.

Per arm and cell: accuracy, mean tokens, rounds run, stop reasons, reviewer accept rate (share
of rounds after which the reviewer accepted), carried-state tokens per round (executor prompt
minus the fixed system+question part), budget left before each extra round, and repairs and
breakages between consecutive rounds. Cells without data are skipped, so this runs on partial
results."""
import json, os, sys, collections, statistics as st
from scipy.stats import binomtest
FAM = {'GLM': ('GLM-5.3-Flash', 'glmst', {24: 'glmx4k', 8: 'glmr4k'}, 'glmx4kn'),
       'DeepSeek': ('DeepSeek-V4.1-Flash', 'dsst', {24: 'dsx4k', 8: 'dsr4k'}, None)}
ARMS = ['ST-Rebuild', 'ST-Accum', 'ST-Window2', 'ST-Summary', 'ST-Slots', 'ST-Slots-M1', 'ST-Rebuild-M1']
R = lambda d: [r for r in d['rounds'] if r['role'] == 'executor']
def load(path, cfg):
    o = {}
    if not path or not os.path.exists(path): return o
    for l in open(path):
        try: d = json.loads(l)
        except: continue
        if d['config'] == cfg: o[d['task_id']] = d
    return o
def cell(fam, t, k):
    tag, st_pre, rb, rbmu = FAM[fam]
    out = {}
    stp = f'traces/tasks_sweep_{tag}_{st_pre}_{t}{k}.jsonl'
    for a in ARMS[1:]: out[a] = load(stp, a)
    rbp = f'traces/tasks_sweep_{tag}_{rb[k]}_{t}{k}.jsonl'
    if t == 'mu' and k == 24 and rbmu: rbp = f'traces/tasks_sweep_{tag}_{rbmu}_mu24.jsonl'
    out['ST-Rebuild'] = load(rbp, 'V5-Unc-Fmt')
    return out
def summarise(D, ids):
    acc = 100 * st.mean(bool(D[i]['success']) for i in ids)
    tok = st.mean(D[i]['total_used'] for i in ids)
    rounds = st.mean(len(R(D[i])) for i in ids)
    stops = collections.Counter(D[i]['stopped_by'] for i in ids)
    rep = dam = trans = 0; carried = []; left = []
    for i in ids:
        d = D[i]; rs = R(d); fx = d.get('fixed_prompt_tokens')
        for j, r in enumerate(rs):
            if fx is not None and j > 0: carried.append(r['prompt_tokens'] - fx)
        for j in range(len(rs) - 1):
            a, b = bool(rs[j]['is_correct']), bool(rs[j+1]['is_correct']); trans += 1
            rep += (not a) and b; dam += a and (not b)
            if rs[j].get('reserve_after') is not None: left.append(rs[j]['reserve_after'])
    return dict(acc=acc, tok=tok, rounds=rounds, budget_stop=100*stops.get('budget', 0)/len(ids),
                trans=trans, net=100*(rep-dam)/trans if trans else float('nan'), rep=rep, dam=dam,
                carried=st.mean(carried) if carried else None, left=st.median(left) if left else None)
rows = []
for fam in FAM:
    for k in (24, 8):
        for t, ds in [('ma','MATH'),('ga','GAIA'),('fr','FRAMES'),('mu','MuSiQue')]:
            C = cell(fam, t, k)
            have = [a for a in ARMS if len(C[a]) >= 30]
            if len(have) < 2: continue
            ids = set.intersection(*[set(C[a]) for a in have])
            if len(ids) < 30: continue
            base = {i: bool(C['ST-Rebuild'][i]['success']) for i in ids} if 'ST-Rebuild' in have else None
            print(f"\n== {fam} {ds} {k}k  (n={len(ids)})")
            print(f"  {'arm':<14}{'acc':>6}{'tok':>8}{'rounds':>7}{'budget%':>8}{'carried':>8}{'left':>7}{'trans':>7}{'rep':>5}{'dam':>5}{'net/100':>8}  vs Rebuild")
            for a in have:
                s = summarise(C[a], ids); s.update(fam=fam, ds=ds, k=k, arm=a, n=len(ids)); rows.append(s)
                cmpx = ''
                if base and a != 'ST-Rebuild':
                    A = {i: bool(C[a][i]['success']) for i in ids}
                    b = sum(1 for i in ids if A[i] and not base[i]); c = sum(1 for i in ids if base[i] and not A[i])
                    cmpx = f"{100*(b-c)/len(ids):+5.1f} p={binomtest(b,b+c,.5).pvalue if b+c else 1:.0e}"
                cr = f"{s['carried']:.0f}" if s['carried'] is not None else '-'
                lf = f"{s['left']/1000:.1f}k" if s['left'] is not None else '-'
                print(f"  {a:<14}{s['acc']:>6.1f}{s['tok']:>8.0f}{s['rounds']:>7.2f}{s['budget_stop']:>8.1f}{cr:>8}{lf:>7}{s['trans']:>7}{s['rep']:>5}{s['dam']:>5}{s['net']:>+8.2f}  {cmpx}")
json.dump(rows, open('results/state_policy.json', 'w'), indent=1)
