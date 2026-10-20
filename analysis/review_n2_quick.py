"""N2 (quick): the earlier live CLIMB runs, re-scored with the corrected scorers. Thresholds were
certified on the calibration halves under the old labels and are kept as run; we also report
whether each would be certified (eps = 5%) under the corrected labels on the same halves.
Per setting, on the test half: live CLIMB vs. the reference pipeline (paired loss, exact McNemar),
token share, stopping frequencies, and the replay prediction for the same threshold and tasks.
Writes paper/data/review_n2_quick.json."""
import json, collections, sys, numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, 'paper/analysis')
from climb_core import load, outcome_matrix, p_betting, GRID, TAU
P = json.load(open('results/climb_live_plan.json')); d = load()
TR = {'GLM': 'GLM-5.3-Flash', 'DeepSeek': 'DeepSeek-V4.1-Flash'}; AB = {'MATH': 'ma', 'FRAMES': 'fr', 'MuSiQue': 'mu'}
out = {}
for key, p in P.items():
    m, b = key.split('|'); t = p['threshold_v3']
    if t == 'direct' or b not in AB: continue
    x = d[(d.model == m) & (d.bench == b)].set_index('task_id')
    live = {json.loads(l)['task_id']: json.loads(l) for l in open(f'traces/tasks_sweep_{TR[m]}_climb3_{AB[b]}24.jsonl')}
    te = [i for i in p['test_ids'] if i in live and i in x.index]; xt = x.loc[te]
    yl = np.array([int(live[i]['success']) for i in te]); cl = np.array([live[i]['total_used'] for i in te], float)
    lv = collections.Counter(live[i]['escalation_level'] for i in te)
    s0 = (xt.S0 > t) & (xt.H0 <= TAU); s1 = ~s0 & (xt.S1 > t) & (xt.H1 <= TAU)
    yr = np.where(s0, xt.Y0, np.where(s1, xt.Y1, xt.Y2)); cr = np.where(s0, xt.C0, np.where(s1, xt.C1, xt.C2))
    yk = xt.YK.to_numpy(); ck = xt.CK.to_numpy(float)
    b01 = int(((yk == 1) & (yl == 0)).sum()); b10 = int(((yk == 0) & (yl == 1)).sum())
    # would the threshold be certified under corrected labels on the calibration half?
    xc = x.loc[[i for i in p['cal_ids'] if i in x.index]].reset_index(drop=True); Y, C, L = outcome_matrix(xc)
    pv = p_betting(L[1:], 0.05, 0.1); j = int(np.argmin(np.abs(np.array(GRID) - t)))
    cert = bool((pv[:j + 1] <= 0.1).all())
    out[key] = dict(n=len(te), threshold=t, ref_acc=100 * yk.mean(), live_acc=100 * yl.mean(), live_loss=100 * (yk.mean() - yl.mean()),
                    p=binomtest(b10, b01 + b10).pvalue if b01 + b10 else 1.0, live_tok=100 * cl.mean() / ck.mean(),
                    replay_acc=100 * yr.mean(), replay_loss=100 * (yk.mean() - yr.mean()), replay_tok=100 * cr.mean() / ck.mean(),
                    live_stop=[100 * lv[k] / len(te) for k in (0, 1, 2)],
                    replay_stop=[100 * s0.mean(), 100 * s1.mean(), 100 * (1 - s0.mean() - s1.mean())],
                    certified_now=cert)
    o = out[key]
    print(f"{key:<17} n={o['n']:4d} t={t:<7} ref {o['ref_acc']:.1f}  live {o['live_acc']:.1f} (loss {o['live_loss']:+.1f}, p={o['p']:.2f}, tok {o['live_tok']:.0f}%)"
          f"  replay {o['replay_acc']:.1f} (loss {o['replay_loss']:+.1f}, tok {o['replay_tok']:.0f}%)  stop live {[round(v) for v in o['live_stop']]} replay {[round(v) for v in o['replay_stop']]}  certified now: {cert}")
json.dump(out, open('paper/data/review_n2_quick.json', 'w'), indent=1)
