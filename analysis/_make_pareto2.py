#!/usr/bin/env python
"""Accuracy-cost frontier, four arms, from traces/ directly.

Replaces the earlier figure, which plotted a different model and a version of
the system that still had the verbalized escalation channel.
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NM = {'ma': 'MATH', 'ga': 'GAIA', 'fr': 'FRAMES', 'mu': 'MuSiQue'}
ARMS = [('V5-Unc-Fmt', 'Naive+Fmt', '#8a8580', 'o', '-'),
        ('Compaction', 'Compaction', '#b4623a', 's', '--'),
        ('V5.1-Ab7-NoVerbal', 'Governed (ours)', '#2e6f6a', 'D', '-')]


def load(path, force=None, into=None):
    into = into if into is not None else {}
    if not os.path.exists(path): return into
    for ln in open(path):
        r = json.loads(ln)
        into.setdefault(force or r['config'], {})[r['task_id']] = r
    return into


fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
plt.rcParams.update({'font.family': 'serif'})
for ax, ds in zip(axes, ('ma', 'ga', 'fr', 'mu')):
    series = {k: [] for k, *_ in ARMS}
    for b in ('8', '16', '24'):
        D = {}
        load(f'traces/tasks_sweep_Qwen3.8-27B_q38_{ds}{b}.jsonl', into=D)
        load(f'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{ds}{b}.jsonl', into=D)
        if b == '24':
            load(f'traces/tasks_sweep_Qwen3.8-27B_q38_ab7_{ds}.jsonl', into=D)
        load(f'traces/tasks_sweep_Qwen3.8-27B_cmp_{ds}{b}k.jsonl', 'Compaction', D)
        if 'V5-Unc-Fmt' not in D: continue
        ids = sorted(set.intersection(*(set(v) for v in D.values())))
        for key, *_ in ARMS:
            if key not in D: continue
            A = D[key]
            series[key].append((sum(A[i]['total_used'] for i in ids)/len(ids),
                                sum(A[i]['success'] for i in ids)/len(ids)*100, b))
    for key, lab, col, mk, lsty in ARMS:
        pts = sorted(series[key])
        if not pts: continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lsty, color=col,
                marker=mk, ms=5, lw=1.6, label=lab, zorder=3 if 'ours' in lab else 2)
    ax.set_title(NM[ds], fontsize=11)
    ax.set_xlabel('mean tokens per task', fontsize=9)
    ax.grid(alpha=.25, lw=.6)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel('success rate (%)', fontsize=9)
axes[0].legend(fontsize=8, frameon=False, loc='lower right')
fig.suptitle('Up and to the left is better. Each line joins one arm across budgets of 8k, 16k and 24k.',
             fontsize=9, y=1.02, color='#55504a')
fig.tight_layout()
fig.savefig('paper/pareto_frontier.pdf', bbox_inches='tight')
print('wrote paper/pareto_frontier.pdf')
for ds in ('ma','ga','fr','mu'):
    pass
