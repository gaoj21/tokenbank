#!/usr/bin/env python
"""Figure: accuracy-cost frontier per dataset. Rerun after new runs land.

Reads paper/ALL_RESULTS.json, so budget points fill in automatically. This is
deliberately not a redraw of any table: it carries a two-dimensional claim (where
each arm sits in the accuracy-cost plane, and whether the arms trace one frontier
or two) that a column of numbers cannot show.

Budget is encoded by marker size rather than a text label on every point, so the
panels stay readable at column width.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': .6,
})

A = json.load(open('paper/ALL_RESULTS.json'))
QW = 'Qwen3.6-27B-FP8'
OURS, FMT, UNC = 'V5.1-NoPool', 'V5-Unc-Fmt', 'V5-Unconstrained'

def cell(tag, cfg):
    v = A.get(tag)
    if not v or v['model'] != QW:
        return None
    c = v['cells'].get(cfg)
    return (c['tok'], c['sr']) if c else None

SRC = {
 'MATH': {
   OURS: {8: ('v55_ma8', OURS), 16: ('v51t16_ma', OURS), 24: ('v51t_ma', OURS)},
   FMT:  {8: ('v55_ma8', FMT),  16: ('v53e1_ma16', FMT), 24: ('v53e1_ma24', FMT)},
   UNC:  {8: ('v55_ma8', UNC),  16: ('v51t16_ma', UNC),  24: (None, None)}},
 'GAIA': {
   OURS: {8: ('v53e3_ga8', OURS), 16: ('v51t16_ga', OURS), 24: ('v51t_ga', OURS)},
   FMT:  {8: ('v53e3_ga8', FMT),  16: ('v52c1_ga16', FMT), 24: ('v52c1_ga24', FMT)},
   UNC:  {8: ('v55_ga8_unc', UNC),16: ('v51t16_ga', UNC),  24: ('v5test_ga', UNC)}},
 'FRAMES': {
   OURS: {8: ('v53e3_fr8', OURS), 16: ('v51t16_fr', OURS), 24: ('v51t_fr', OURS)},
   FMT:  {8: ('v53e3_fr8', FMT),  16: ('v52c1_fr16', FMT), 24: ('v52c1_fr24', FMT)},
   UNC:  {8: ('v55_fr8_unc', UNC),16: ('v51t16_fr', UNC),  24: (None, None)}},
 'MuSiQue': {
   OURS: {8: ('v55_mu8', OURS), 16: ('v51t16_mu', OURS), 24: ('v51t_mu', OURS)},
   FMT:  {8: ('v55_mu8', FMT),  16: ('v55_mu16_fmt', FMT), 24: ('v53e2b_mu_fmt', FMT)},
   UNC:  {8: ('v55_mu8', UNC),  16: ('v51t16_mu', UNC),  24: ('v5test_mu', UNC)}},
}
HARD = {('MATH', UNC, 24): (6333, 52.8), ('FRAMES', UNC, 24): (8251, 10.6)}

STYLE = {OURS: ('#17546B', 'Ours'),
         FMT:  ('#BE6A33', 'Naive+Fmt'),
         UNC:  ('#9AA0A8', 'Naive')}
SIZE = {8: 20, 16: 40, 24: 70}          # budget encoded by marker area
VERDICT = {
 'MATH':    'separated frontiers',
 'GAIA':    'one frontier',
 'FRAMES':  'one frontier',
 'MuSiQue': 'same accuracy, fewer tokens',
}

fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0))
for k, (ax, ds) in enumerate(zip(axes.ravel(), SRC)):
    for arm, (col, lab) in STYLE.items():
        pts = []
        for b in (8, 16, 24):
            p = HARD.get((ds, arm, b))
            if p is None:
                tag, cfg = SRC[ds][arm].get(b, (None, None))
                p = cell(tag, cfg) if tag else None
            if p:
                pts.append((b, p[0], p[1]))
        if not pts:
            continue
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
        ax.plot(xs, ys, color=col, lw=1.0, alpha=.55, zorder=2, solid_capstyle='round')
        ax.scatter(xs, ys, s=[SIZE[p[0]] for p in pts], color=col,
                   zorder=3, edgecolor='white', linewidth=.7)
        # direct label at the largest-budget end, no legend
        bx, by = xs[-1], ys[-1]
        ax.annotate(lab, (bx, by), textcoords='offset points', xytext=(7, -1),
                    fontsize=6.8, color=col, va='center')
    ax.set_title(f'{ds}', fontsize=9.5, pad=9, loc='left')
    ax.text(0, 1.015, VERDICT[ds], transform=ax.transAxes, fontsize=6.8,
            color='#6B7280', va='bottom', style='italic')
    ax.grid(alpha=.16, lw=.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=6.8, length=2.5, width=.5)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f'{v/1000:g}k' if v >= 1000 else f'{v:g}'))
    ax.margins(x=.22, y=.20)
    if k >= 2:
        ax.set_xlabel('mean tokens per task', fontsize=7.5)
    if k % 2 == 0:
        ax.set_ylabel('success rate (\\%)', fontsize=7.5)

# one shared key for the budget encoding
for b, x in zip((8, 16, 24), (.40, .455, .515)):
    fig.scatter = fig.text(x, -.005, '', fontsize=1)
ax0 = fig.add_axes([.38, -.045, .26, .04]); ax0.axis('off')
for i, b in enumerate((8, 16, 24)):
    ax0.scatter([i * .28 + .05], [.5], s=SIZE[b], color='#4B5563',
                edgecolor='white', linewidth=.7, transform=ax0.transAxes)
    ax0.text(i * .28 + .11, .5, f'{b}k', fontsize=6.8, va='center',
             transform=ax0.transAxes, color='#374151')
ax0.text(-.02, .5, 'budget', fontsize=6.8, va='center', ha='right',
         transform=ax0.transAxes, color='#374151')

fig.tight_layout(h_pad=2.4, w_pad=2.6)
fig.savefig('paper/pareto_frontier.pdf', bbox_inches='tight')
print('paper/pareto_frontier.pdf written')
for ds in SRC:
    have = sum(1 for arm in STYLE for b in (8, 16, 24)
               if HARD.get((ds, arm, b)) or
               (SRC[ds][arm].get(b, (None,))[0] and cell(*SRC[ds][arm][b])))
    print(f'  {ds:8} {have}/9 points')
