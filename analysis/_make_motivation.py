#!/usr/bin/env python
"""Opening figure: what a budgeted loop spends, next to what that spending buys.

The point of putting both panels side by side is that the ordering reverses
between them. The plan is the largest thing the pipeline carries and the only
one that can be removed for free; the critique is a twentieth of its size and
cannot.
"""
import json, glob, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEG = [('plan', 'plan', '#b4623a'), ('question', 'question', '#7d9aa6'),
       ('prior_answer', 'working draft', '#c8bfae'), ('system', 'system', '#e0d8c8'),
       ('critique', 'critique', '#2e6f6a'), ('tool_output', 'tool output', '#4a4540')]
ORDER = ['GAIA', 'MATH', 'FRAMES', 'MUSIQUE', 'BCB']
LABEL = {'MUSIQUE': 'MuSiQue', 'BCB': 'BigCodeBench'}

shares, names = [], []
for ds in ORDER:
    f = f'traces/tasks_sweep_Qwen3.8-27B_seg_{ds}.jsonl'
    agg = collections.Counter(); n = 0
    for ln in open(f):
        sc = (json.loads(ln).get('segment_chars') or {})
        if not sc: continue
        n += 1
        agg.update(sc)
    agg.pop('_completion', None)
    t = sum(agg.values())
    shares.append([agg.get(k, 0)/t*100 for k, _, _ in SEG])
    names.append(LABEL.get(ds, ds))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 3.6),
                               gridspec_kw={'width_ratios': [2.0, 1.0]})
plt.rcParams.update({'font.family': 'serif'})

y = np.arange(len(names))
left = np.zeros(len(names))
S = np.array(shares)
for j, (key, lab, col) in enumerate(SEG):
    axL.barh(y, S[:, j], left=left, color=col, height=.62,
             label=lab, edgecolor='white', linewidth=.7)
    left += S[:, j]
axL.set_yticks(y); axL.set_yticklabels(names, fontsize=9)
axL.invert_yaxis(); axL.set_xlim(0, 100)
axL.set_xlabel('share of every character the pipeline processes (%)', fontsize=9)
axL.legend(fontsize=8, frameon=False, ncol=6, loc='upper center',
           bbox_to_anchor=(.5, 1.22))
axL.tick_params(labelsize=8)
for i in range(len(names)):
    axL.text(S[i, 0]/2, i, f'{S[i,0]:.0f}%', ha='center', va='center',
             fontsize=8.5, color='white', fontweight='bold')

# right: what removing each slot costs, pooled over 850 paired items
bars = [('plan\n$\\to$ current step', 8, 5, 0.581, '#b4623a'),
        ('critique\n$\\to$ verdict', 31, 13, 0.010, '#2e6f6a')]
for i, (lab, cw, wc, p, col) in enumerate(bars):
    axR.barh(i, -cw, color=col, height=.45, alpha=.85)
    axR.barh(i, wc, color=col, height=.45, alpha=.35)
    axR.text(0, i - .42, f'$p={p}$', ha='center', fontsize=9,
             color='#1b1a18' if p < .05 else '#6f6b64',
             fontweight='bold' if p < .05 else 'normal')
axR.axvline(0, color='#55504a', lw=.9)
axR.set_yticks(range(len(bars)))
axR.set_yticklabels([b[0] for b in bars], fontsize=9)
axR.invert_yaxis()
axR.set_xlabel('answers broken $\\leftarrow$   |   $\\rightarrow$ answers fixed', fontsize=9)
axR.set_title('what bounding it costs (850 paired items)', fontsize=9.5)
axR.tick_params(labelsize=8)
axR.spines[['top', 'right']].set_visible(False)
axL.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig('paper/motivation.pdf', bbox_inches='tight')
print('wrote paper/motivation.pdf')
print('plan share per benchmark:', ' '.join(f'{n}={s[0]:.1f}%' for n, s in zip(names, shares)))
