import re, glob, builtins, os as _os
HAND = {'tab_main', 'tab_calib', 'tab_iid', 'tab_diagnostics'}   # hand-formatted in the draft; numbers updated by hand
_open = builtins.open
def open(path, mode='r', *a, **k):
    if 'w' in mode and _os.path.basename(str(path)).replace('.tex', '') in HAND:
        return _open(_os.devnull, mode, *a, **k)
    return _open(path, mode, *a, **k)
"""Generate every results table of the paper from the result files (run from the repo root)."""
import json
M = json.load(open('results/main_table_final.json'))
A = json.load(open('results/ablations_glm.json'))
X = json.load(open('results/appendix_glm.json'))
MG = json.load(open('results/monotone_gain.json'))
CELLS = [('GLM', b) for b in ('MATH', 'GAIA', 'FRAMES', 'MuSiQue')] + [('DeepSeek', b) for b in ('MATH', 'GAIA', 'FRAMES', 'MuSiQue')]
mn = lambda f: 'GLM' if f == 'GLM' else 'DeepSeek'
def at(acc, tok, bold_tok=False, bold_acc=False):
    a = f'\\textbf{{{acc:.1f}}}' if bold_acc else f'{acc:.1f}'
    t = f'\\textbf{{{tok:.0f}\\%}}' if bold_tok else f'{tok:.0f}\\%'
    return f'{a} / {t}'
out = {}

# ---------------- main table (pipeline vs CLIMB) and baselines (appendix) ----------------
rows, brows = [], []
for i, (f, b) in enumerate(CELLS):
    r = M[f'{f}|{b}']; pa = r['pipe_acc']
    cells = [f"{pa:.1f} / {r['pipe_tok']/1000:.1f}k"]
    for e in ('eps2', 'eps5', 'eps10'):
        cells.append(at(r[e]['acc'], r[e]['tok'], bold_tok=r[e]['tok'] < 50, bold_acc=r[e]['acc'] >= pa))
    viol = max(r[e]['viol_pool'] for e in r if e.startswith('eps'))
    lead = (f'\\multirow{{4}}{{*}}{{{mn(f)}}}' if i % 4 == 0 else '')
    sep = '\n\\midrule' if i == 3 else ''
    rows.append(f"{lead} & {b} & " + ' & '.join(cells) + f" & {viol:.1f}\\% \\\\" + sep)
    bcells = [f"{pa:.1f} / 100\\%"] + [at(r[k][0], r[k][1]) for k in ('L0', 'L1', 'fixed', 'oracle')] + [at(r['eps5']['acc'], r['eps5']['tok'])]
    brows.append(f"{lead} & {b} & " + ' & '.join(bcells) + " \\\\" + sep)
out['tab_main'] = r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}llcccccc@{}}
\toprule
& & & \multicolumn{3}{c}{\climb{}} & \\
\cmidrule(lr){4-6}
Model & Benchmark & Pipeline & $\eps = 2\%$ & $\eps = 5\%$ & $\eps = 10\%$ & Viol. \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Replay results: accuracy / tokens as a share of the pipeline's (pipeline: accuracy / mean tokens), averaged over
200 calibration/evaluation splits. Viol.\ is the largest share of splits, over the three tolerances, in which the chosen
policy's empirical loss on the full recorded pool exceeds $\eps$. This diagnostic is not itself the population-risk
event controlled at $\delta = 10\%$. Bold: under 50\% of the
pipeline's tokens, or at least as accurate as the pipeline. Uncertified baselines are in Table~\ref{tab:baselines}.}
\label{tab:main}
\end{table*}
"""
out['tab_baselines'] = r"""\begin{table*}[h]
\centering
\small
\begin{tabular}{@{}llcccccc@{}}
\toprule
Model & Benchmark & Pipeline & Always $L_0$ & Always $L_1$ & Fixed $t = -0.10$ & Oracle & \climb{} ($\eps = 5\%$) \\
\midrule
""" + '\n'.join(brows) + r"""
\bottomrule
\end{tabular}
\caption{Uncertified baselines on the full test split, accuracy / tokens as a share of the pipeline's. Always $L_k$ stops
every task at rung $k$; Oracle stops at the cheapest rung that answers correctly. None of these carries a guarantee;
\climb{} is repeated from Table~\ref{tab:main}.}
\label{tab:baselines}
\end{table*}
"""

# ---------------- calibration design (GLM) ----------------
T2 = json.load(open('paper/data/review_table2.json'))      # parametrised re-run (analysis/review_table2.py)
names = [(r, r) for r in T2['rows']]
C = {b: {r: [round(v[0]), v[1]] for r, v in T2['res'][b].items()} for b in T2['res']}; rows = []
for k, nm in names:
    rows.append(nm + ' & ' + ' & '.join(f"{C[b][k][0]}\\%" for b in ('MATH', 'GAIA', 'FRAMES', 'MuSiQue')) + ' \\\\')
out['tab_calib'] = r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lrrrr@{}}
\toprule
Test (GLM, $\eps{=}5\%$) & MATH & GAIA & FRAMES & MuSiQue \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Tokens (\% of pipeline) under different certification tests and procedures (fixed-sequence testing unless marked
Bonferroni; a union of two tests splits $\delta$ equally), same 200 splits; every row keeps the empirical finite-pool
violation rate below the nominal level $\delta$ (Table~\ref{tab:calib_full}).}
\label{tab:calib}
\end{table}
"""

# ---------------- appendix: monotone ----------------
rows = [f"{r['fam'].replace('DeepSeek','DS')} & {r['ds']} & $L_{r['rung'][1]}$ & " + ' & '.join(f'${100*v:+.1f}$' for v in r['gain']) + f" & ${r['rho']:+.2f}$ \\\\" for r in MG]
out['tab_monotone'] = r"""\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lllrrrrrr@{}}
\toprule
Model & Bench. & Rung & Q1 & Q2 & Q3 & Q4 & Q5 & $\rho$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Accuracy gain from escalating to the pipeline (points) by confidence quintile (Q1 least confident), and Spearman
$\rho$ between quintile and gain. $\rho \le 0$ in """ + str(sum(r['rho'] <= 0 for r in MG)) + r""" of 16 rows.}
\label{tab:monotone}
\end{table}
"""

# ---------------- appendix: delta / n / shift ----------------
B = ('MATH', 'GAIA', 'FRAMES', 'MuSiQue')
rows = [f"$\\delta = {d}$ & " + ' & '.join(f"{X['delta'][b][d][0]}\\% / {X['delta'][b][d][1]:.0f}\\%" for b in B) + ' \\\\' for d in ('0.05', '0.1', '0.2')]
rows.append('\\midrule')
for n in ('50', '100', '200'):
    rows.append(f"$n = {n}$ & " + ' & '.join((f"{X['n'][b][n][0]}\\% / {X['n'][b][n][1]:.0f}\\%" if n in X['n'][b] else '--') for b in B) + ' \\\\')
out['tab_sens'] = r"""\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
GLM, $\eps = 5\%$ & MATH & GAIA & FRAMES & MuSiQue \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Sensitivity: tokens (\% of pipeline) / violation rate, as the error level $\delta$ and the calibration size $n$ vary
(default $\delta = 0.1$, $n$ = half of the test split).}
\label{tab:sens}
\end{table}
"""
rows = []
for src in B:
    rows.append(f"{src} & " + ' & '.join(('--' if src == dst else f"{X['shift'][src][dst]['tok']}\\% / ${X['shift'][src][dst]['risk']:+.1f}$") for dst in B) + ' \\\\')
out['tab_shift'] = r"""\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccc@{}}
\toprule
Calibrated on $\downarrow$ / deployed on $\rightarrow$ & MATH & GAIA & FRAMES & MuSiQue \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Distribution shift (GLM, $\eps = 5\%$): tokens / risk (points) when the threshold calibrated on one benchmark is
deployed on another. Risk above 5 violates the tolerance.}
\label{tab:shift}
\end{table}
"""
for k, v in out.items():
    open(f'paper/sections/{k}.tex', 'w').write(v)
print('wrote', list(out))

# ---------------- appendix: single call (4k cap) and generation caps ----------------
import statistics as _st
def _single(path):
    L = [json.loads(l) for l in open(path)]
    return 100 * _st.mean(x['success'] for x in L), _st.mean(x['total_used'] for x in L)
SINGLE = {'GLM|MATH': 'traces/tasks_sweep_GLM-5.3-Flash_glms4k_ma24.jsonl', 'GLM|GAIA': 'traces/tasks_sweep_GLM-5.3-Flash_glms4k_ga24.jsonl',
          'GLM|FRAMES': 'traces/tasks_sweep_GLM-5.3-Flash_glms4k_fr24.jsonl', 'GLM|MuSiQue': 'traces/tasks_sweep_GLM-5.3-Flash_glmfp8s_mu24.jsonl',
          'DeepSeek|MATH': 'traces/tasks_sweep_DeepSeek-V4.1-Flash_dss4k_ma24.jsonl', 'DeepSeek|GAIA': 'traces/tasks_sweep_DeepSeek-V4.1-Flash_dss4k_ga24.jsonl',
          'DeepSeek|FRAMES': 'traces/tasks_sweep_DeepSeek-V4.1-Flash_dss4k_fr24.jsonl', 'DeepSeek|MuSiQue': 'traces/tasks_sweep_DeepSeek-V4.1-Flash_dss4k_mu24.jsonl'}
rows = []
for k, pth in SINGLE.items():
    a, t = _single(pth); r = M[k]
    rows.append(f"{k.replace('DeepSeek','DS').replace('|',' ')} & {a:.1f} / {t/1000:.1f}k & {r['eps5']['acc']:.1f} & {r['pipe_acc']:.1f} / {r['pipe_tok']/1000:.1f}k \\\\")
open('paper/sections/tab_single.tex', 'w').write(r"""\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
& Single call & \climb{} ($\eps{=}5\%$) & Pipeline \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{One executor call with default reasoning and the same 4{,}096 cap, for reference.}
\label{tab:single}
\end{table}
""")
CAPS = json.load(open('results/caps_table.json'))
fmt = lambda v: f"${v[0]:+.1f}" + ("^{*}$" if v[1] else "$")
rows = [f"{r['cell']} & {fmt(r['24'])} & {fmt(r['16'])} & {fmt(r['8'])} \\\\" for r in CAPS]
open('paper/sections/tab_caps.tex', 'w').write(r"""\begin{table}[h]
\centering
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
& 24k & 16k & 8k \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Accuracy change of the plain pipeline from a 1{,}024 to a 4{,}096 per-call cap, by task
budget. $^{*}$: $p < 0.05$, exact McNemar.}
\label{tab:caps}
\end{table}
""")
print('wrote tab_single, tab_caps')

# ---------------- appendix: expanded tables ----------------
RUNG = json.load(open('results/rung_table.json'))
rows = []
for i, (f, b) in enumerate(CELLS):
    r = RUNG[f'{f}|{b}']; lead = (f'\\multirow{{4}}{{*}}{{{mn(f)}}}' if i % 4 == 0 else '')
    cells = [f"{r[k][0]:.1f} / {r[k][1]/1000:.1f}k" for k in ('L0', 'L1', 'S', 'P')]
    rows.append(f"{lead} & {b} & " + ' & '.join(cells) + f" & {r['auc0']:.2f} & {r['auc1']:.2f} & {r['rescue']:.1f} & {r['harm']:.1f} \\\\" + ('\n\\midrule' if i == 3 else ''))
open('paper/sections/tab_rungs.tex', 'w').write(r"""\begin{table*}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llccccccrr@{}}
\toprule
Model & Bench. & $L_0$ & $L_1$ (cum.) & $L_2$ seeded (cum.) & Pipeline & AUROC $L_0$ & AUROC $L_1$ & Rescue & Harm \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Per-rung accuracy / mean cumulative tokens on the full test split, from the deployed-configuration records.
AUROC of each rung's confidence for its own correctness. Rescue: share of tasks the pipeline answers correctly and $L_0$
does not; Harm: the reverse (\%).}
\label{tab:rungs}
\end{table*}
""")
ROLE = {}
for line in open('results/role_shares.txt'):
    m = re.match(r'\s+(GLM|DeepSeek)\s+(\S+)\s+planner\s+([\d.]+)%\s+executor\s+([\d.]+)%\s+reviewer\s+([\d.]+)%', line)
    if m: ROLE[(m.group(1), m.group(2))] = tuple(float(x) for x in m.group(3, 4, 5))
rows = [f"{mn(f)} & {b} & {ROLE[(f, b)][0]:.1f} & {ROLE[(f, b)][1]:.1f} & {ROLE[(f, b)][2]:.1f} & {ROLE[(f, b)][0] + ROLE[(f, b)][2]:.1f} \\\\"
        for f in ('GLM', 'DeepSeek') for b in ('MATH', 'GAIA', 'FRAMES', 'MuSiQue', 'pooled')]
open('paper/sections/tab_roles.tex', 'w').write(r"""\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}llrrrr@{}}
\toprule
Model & Bench. & Planner & Executor & Reviewer & Coord. \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Share (\%) of the plain pipeline's tokens by role (24k budget, 4{,}096 cap). Coord.: planner + reviewer.}
\label{tab:roles}
\end{table}
""")
SWP = {}
for pth in glob.glob('results/eps_sweep_*.json'):
    d = json.load(open(pth)); SWP[(d['fam'], d['ds'])] = d
EPSL = ['0.01', '0.02', '0.03', '0.05', '0.075', '0.1']
rows = []
for i, (f, b) in enumerate(CELLS):
    d = SWP[(f, b)]; lead = (f'\\multirow{{4}}{{*}}{{{mn(f)}}}' if i % 4 == 0 else '')
    rows.append(f"{lead} & {b} & " + ' & '.join(f"{d['sweep'][e]['tok']:.0f} / {d['pipe_acc'] - d['sweep'][e]['acc']:.1f}" for e in EPSL) + " \\\\" + ('\n\\midrule' if i == 3 else ''))
open('paper/sections/tab_sweep.tex', 'w').write(r"""\begin{table*}[h]
\centering
\small
\begin{tabular}{@{}llcccccc@{}}
\toprule
Model & Bench. & $\eps = 1\%$ & $2\%$ & $3\%$ & $5\%$ & $7.5\%$ & $10\%$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Tolerance sweep: tokens (\% of pipeline) / accuracy loss (points) of \climb{}, 100 calibration/evaluation splits per
point. Violation rates are at most """ + f"{max(v['viol'] for d in SWP.values() for v in d['sweep'].values()):.0f}" + r"""\% at every point.}
\label{tab:sweep}
\end{table*}
""")
POL = A['policies']; PV = A.get('policy_violations_eval', {})
rows = [f"{b} & " + ' & '.join(f"{POL[b][m]:.1f}" for m in 'ABCDE') + " \\\\" for b in ('MATH', 'GAIA', 'FRAMES', 'MuSiQue')]
open('paper/sections/tab_policies.tex', 'w').write(r"""\begin{table}[h]
\centering
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
GLM & A & B & C & D & E \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Tokens (\% of pipeline) at $\eps = 5\%$ for the five escalation policies of \S\ref{sec:variants} (200 splits,
40/20/40). A: shared threshold (\climb{}); B: separate thresholds; C: answer agreement; D: Bayes deciles; E: learned values.}
\label{tab:policies}
\end{table}
""")
rows = []
for k, nm in names:
    rows.append(nm + ' & ' + ' & '.join(f"{C[b][k][0]} / {C[b][k][1]:.1f}" for b in ('MATH', 'GAIA', 'FRAMES', 'MuSiQue')) + ' \\\\')
open('paper/sections/tab_calib_full.tex', 'w').write(r"""\begin{table}[h]
\centering
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccc@{}}
\toprule
Test (GLM, $\eps{=}5\%$) & MATH & GAIA & FRAMES & MuSiQue \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Tokens (\% of pipeline) / violation rate (\%) for each certification test (200 splits).}
\label{tab:calib_full}
\end{table}
""")
print('wrote tab_rungs, tab_roles, tab_sweep, tab_policies, tab_calib_full')

# ---------------- appendix: i.i.d. guarantee check and (A1) test ----------------
G = json.load(open('paper/data/guarantee_iid.json'))
rows = []
for i, (f, b) in enumerate(CELLS):
    r = G[f'{f}|{b}']; lead = (f'\\multirow{{4}}{{*}}{{{mn(f)}}}' if i % 4 == 0 else '')
    cells = [f"{100*o['viol']:.1f} {{\\scriptsize[{100*o['ci'][0]:.1f}, {100*o['ci'][1]:.1f}]}} & {100*o['tok']:.0f}"
             for o in (r['out'][e] for e in ('0.02', '0.05', '0.1'))]
    rows.append(f"{lead} & {b} & {r['n']} & " + ' & '.join(cells) + " \\\\" + ('\n\\midrule' if i == 3 else ''))
open('paper/sections/tab_iid.tex', 'w').write(r"""\begin{table*}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llrcrcrcr@{}}
\toprule
& & & \multicolumn{2}{c}{$\eps = 2\%$} & \multicolumn{2}{c}{$\eps = 5\%$} & \multicolumn{2}{c}{$\eps = 10\%$} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Model & Bench. & $n$ & $\Pr(R > \eps)$ [95\% CI] & Tok. & $\Pr(R > \eps)$ [95\% CI] & Tok. & $\Pr(R > \eps)$ [95\% CI] & Tok. \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Direct check of Proposition~\ref{prop:risk}. The task distribution is the recorded pool; $n$ calibration tasks
(half the pool) are drawn i.i.d.\ from it, \climb{} is calibrated ($\delta = 10\%$), and the deployed policy's risk under
that distribution is computed exactly. $\Pr(R > \eps)$: share of 5{,}000 draws in which it exceeds $\eps$ (\%), with a
Clopper--Pearson interval; Tok.: mean tokens (\% of pipeline).}
\label{tab:iid}
\end{table*}
""")
A1 = json.load(open('paper/data/a1_sufficiency.json'))
def a1cell(o, auc=True):
    s = f"${1000*o['gain']:+.1f}$ {{\\scriptsize[{1000*o['ci'][0]:+.1f}, {1000*o['ci'][1]:+.1f}]}}"
    if auc: s += f" & ${o['dauc']:+.3f}$ {{\\scriptsize[{o['dauc_ci'][0]:+.2f}, {o['dauc_ci'][1]:+.2f}]}}"
    return s.replace('+-', '-')
rows = []
for i, (f, b) in enumerate(CELLS):
    r = A1[f'{f}|{b}|S']; lead = (f'\\multirow{{4}}{{*}}{{{mn(f)}}}' if i % 4 == 0 else '')
    rows.append(f"{lead} & {b} & " + ' & '.join(a1cell(r['out'][t]) for t in ('Y1', 'Y2', 'YK')) + ' & ' + a1cell(r['out']['logdC'], False)
                + " \\\\" + ('\n\\midrule' if i == 3 else ''))
open('paper/sections/tab_a1.tex', 'w').write(r"""\begin{table*}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llccccccc@{}}
\toprule
& & \multicolumn{2}{c}{$Y_1$ (stop at $L_1$)} & \multicolumn{2}{c}{$Y_2$ (seeded team)} & \multicolumn{2}{c}{$Y_{\mathrm{ref}}$ (reference)} & $\log(C_2 - C_1)$ \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}
Model & Bench. & $\Delta$log-loss & $\Delta$AUC & $\Delta$log-loss & $\Delta$AUC & $\Delta$log-loss & $\Delta$AUC & $\Delta$MSE \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Test of (A1) at $L_1$: held-out improvement from adding the earlier score $S_0$ to a model that sees $S_1$
(cross-fitted, $5 \times 10$ folds; log-loss and MSE reductions $\times 10^{-3}$, positive = history helps; 95\% bootstrap
intervals over tasks).}
\label{tab:a1}
\end{table*}
""")
print('wrote tab_iid, tab_a1')

# ---------------- review analyses (paper/analysis/review_*.py) ----------------
RV = lambda n: json.load(open(f'paper/data/review_{n}.json'))
def tabwrap(label, cols, head, body, cap, star=False, size='\\small'):
    env = 'table*' if star else 'table'
    open(f'paper/sections/{label}.tex', 'w').write(f"""\\begin{{{env}}}[h]
\\centering
{size}
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\resizebox{{{'\\textwidth' if star else '\\columnwidth'}}}{{!}}{{%
\\begin{{tabular}}{{{cols}}}
\\toprule
{head} \\\\
\\midrule
""" + '\n'.join(body) + f"""
\\bottomrule
\\end{{tabular}}}}
\\caption{{{cap}}}
\\label{{tab:{label[4:]}}}
\\end{{{env}}}
""")
SK = [f'{f}|{b}' for f, b in CELLS]; nm = lambda k: f"{mn(k.split('|')[0])} {k.split('|')[1]}"
R1 = RV('r1')
tabwrap('tab_r1', '@{}lrrrrrrr@{}', 'Setting & Ref. & Seeded & $\\Delta$ & $p$ & Disagr. & $C_K$ & $C_2$ ($C_2{-}C_1$)',
        [f"{nm(k)} & {r['accK']:.1f} & {r['acc2']:.1f} & ${r['diff']:+.1f}$ & {r['p']:.2f} & {r['disagree']:.1f} & {r['CK']/1000:.1f}k & {r['C2']/1000:.1f}k ({r['team']/1000:.1f}k) \\\\" for k, r in R1.items()],
        'The two terminal configurations. Ref.: the reference pipeline (no draft), against which the loss is measured; '
        'Seeded: $L_2$ given the cheap rungs\' draft, which \\climb{} runs when it escalates. Accuracy (\\%), paired difference, '
        'exact McNemar $p$, share of tasks on which their correctness differs, mean tokens of the reference and cumulative tokens of the '
        'seeded path (in parentheses, the team alone).')
R2 = RV('r2')['replay']
tabwrap('tab_r2', '@{}lcccccc@{}', 'Setting & \\multicolumn{3}{c}{$\\eps = 5\\%$} & \\multicolumn{3}{c}{$\\eps = 10\\%$} \\\\\n& Det. & Mix. & Untied & Det. & Mix. & Untied',
        [f"{nm(k)} & " + ' & '.join(f"{100*R2[k][e][v]['tok']:.0f}" for e in ('0.05', '0.1') for v in ('det', 'mix', 'untied')) + ' \\\\' for k in SK],
        'Tokens (\\% of the reference) of the deterministic tied threshold (\\climb{}), with randomised mixtures of adjacent thresholds '
        '($q \\in \\{0.25, 0.5, 0.75\\}$) inserted into the fixed sequence, and with untied thresholds $t_1 = t_0 + o$, '
        '$o \\in \\{0, \\pm 0.05, \\pm 0.1\\}$ (one sequence per offset at $\\delta/5$). 200 splits; all violation rates $\\le$ '
        + f"{max(100*R2[k][e][v]['viol'] for k in SK for e in ('0.05','0.1') for v in ('det','mix','untied')):.1f}" + '\\%.')
R3 = RV('r3')
tabwrap('tab_r3', '@{}lcccc@{}', 'Setting & $n$/bin & $\\rho$, $L_0$ [95\\% CI] & $\\rho$, $L_1$ [95\\% CI] & Gain, low $\\to$ high $S_0$',
        [f"{nm(k)} & {min(R3[k]['L0']['n'])} & ${R3[k]['L0']['rho']:+.2f}$ {{\\scriptsize[{R3[k]['L0']['rho_ci'][0]:+.2f}, {R3[k]['L0']['rho_ci'][1]:+.2f}]}} & "
         f"${R3[k]['L1']['rho']:+.2f}$ {{\\scriptsize[{R3[k]['L1']['rho_ci'][0]:+.2f}, {R3[k]['L1']['rho_ci'][1]:+.2f}]}} & "
         f"{R3[k]['L0']['gain'][0]:.1f} {{\\scriptsize[{R3[k]['L0']['lo'][0]:.1f}, {R3[k]['L0']['hi'][0]:.1f}]}} $\\to$ {R3[k]['L0']['gain'][4]:.1f} {{\\scriptsize[{R3[k]['L0']['lo'][4]:.1f}, {R3[k]['L0']['hi'][4]:.1f}]}} \\\\" for k in SK],
        'Uncertainty of the monotone-gain diagnostic. Spearman $\\rho$ between confidence quintile and the gain '
        '$\\E[Y_{\\mathrm{ref}} - Y_k \\mid \\text{quintile}]$, with 95\\% bootstrap intervals (2{,}000 task resamples), and the gain (points) in the '
        'lowest and highest $S_0$ quintile.'.replace('+-', '-'), star=True)
R4 = RV('r4'); BL = [('always_L0', 'Always $L_0$'), ('always_L1', 'Always $L_1$'), ('mix_L0', 'Mix $L_0$/ref.'), ('mix_L1', 'Mix $L_1$/ref.'), ('two_level_L0_ref', '$L_0 \\to$ ref.'), ('climb', '\\climb{}')]
for e, lab in (('0.05', 'tab_r4a'), ('0.1', 'tab_r4b')):
    body = []
    for k in SK:
        best = min(R4[k][e][b]['tok'] for b, _ in BL)
        body.append(f"{nm(k)} & " + ' & '.join((lambda v: f"\\textbf{{{v}}}" if R4[k][e][b]['tok'] == best else v)(f"{100*R4[k][e][b]['tok']:.0f} ({100*R4[k][e][b]['certified']:.0f})") for b, _ in BL) + ' \\\\')
    tabwrap(lab, '@{}l' + 'c' * len(BL) + '@{}', f'$\\eps = {100*float(e):.0f}\\%$ & ' + ' & '.join(n for _, n in BL), body,
            f'Simple policies certified by the same procedure (betting test, fixed sequence, fallback to the reference), $\\eps = {100*float(e):.0f}\\%$: '
            'tokens (\\% of the reference) and, in parentheses, the share of the 200 splits in which a non-trivial policy was certified. '
            'Mix: each task independently goes to the cheap rung with probability $p \\in \\{0.1, \\dots, 0.9\\}$ (increasing $p$ tested '
            'in order), otherwise to the reference. $L_0 \\to$ ref.: a one-rung ladder that escalates to the reference pipeline. '
            'Lowest cost in bold; all violation rates $\\le$ ' + f"{max(100*R4[k][e][b]['viol'] for k in SK for b, _ in BL):.1f}" + '\\%.', star=True)
R4D = RV('r4diag')
tabwrap('tab_r4diag', '@{}lcccccc@{}', 'Setting & \\multicolumn{3}{c}{$\\eps = 5\\%$} & \\multicolumn{3}{c}{$\\eps = 10\\%$} \\\\\n& Fine & Coarse & Free esc. & Fine & Coarse & Free esc.',
        [f"{nm(k)} & " + ' & '.join(f"{100*R4D[k][e][v]['tok']:.0f}" for e in ('0.05', '0.1') for v in ('fine', 'coarse', 'free_escalation')) + ' \\\\' for k in SK],
        'Where the gap to input-agnostic mixing comes from. Tokens of \\climb{} with its 169-point grid (Fine), a 12-point grid '
        '(Coarse), and in a counterfactual in which an escalated task pays only the reference pipeline (Free esc.: no sunk cost '
        'of the failed cheap attempts and no seeding overhead).')
R5 = RV('r5')
tabwrap('tab_r5', '@{}lrrrrrrr@{}', 'Setting & $n_{\\text{cal}}$ & $T_{\\text{cal}}$ & $\\Delta_{5\\%}$ & Payback & $\\Delta_{10\\%}$ & Payback & Calls',
        [f"{nm(k)} & {r['n_cal']} & {r['T_cal']/1e6:.1f}M & {r['0.05']['saving']:.0f} & " + (f"{r['0.05']['payback']:,.0f}" if r['0.05']['payback'] else '--').replace(',', '{,}')
         + f" & {r['0.1']['saving']:.0f} & " + (f"{r['0.1']['payback']:,.0f}" if r['0.1']['payback'] else '--').replace(',', '{,}') + f" & {r['0.1']['calls']:.1f} / {r['calls_ref']:.1f}" + ' \\\\' for k, r in R5.items()],
        'Calibration cost and payback. $T_{\\text{cal}}$: tokens to record a calibration set ($n_{\\text{cal}}$ tasks, each run '
        'through all rungs and the reference); $\\Delta$: mean tokens saved per deployed query at the given tolerance; payback: '
        '$T_{\\text{cal}} / \\Delta$ queries (judge calls excluded). Calls: estimated model calls per query of \\climb{} at $\\eps = 10\\%$ / of '
        'the reference (planner once, executor and reviewer per round, tool follow-ups excluded; the traces log executor rounds only).')
LB = RV('labels')
tabwrap('tab_labels', '@{}lrrrrcccc@{}', 'Setting & \\multicolumn{4}{c}{Changed by rescoring (\\%)} & \\multicolumn{4}{c}{Tokens / viol.\\ at $\\eps = 5\\%$} \\\\\n& $L_0$ & $L_1$ & $L_2$ & Ref. & Current & Worst & Best & Complete',
        [f"{nm(k)} & " + ' & '.join(f"{r['F'][j]:.1f}" for j in '012K') + ' & '
         + ' & '.join(f"{100*r[h]['0.05']['tok']:.0f} / {100*r[h]['0.05']['viol']:.1f}" for h in ('current', 'worst', 'best', 'complete')) + ' \\\\' for k, r in LB.items()],
        'Label quality. Left: share of labels the corrected scorers changed from incorrect to correct. Right: the main replay '
        'under the labels as used (Current); with labels that could not be re-scored (stored output cut before the answer, '
        'all originally incorrect) set in the direction that maximises the loss (Worst: reference correct) or minimises it '
        '(Best: cheap rungs correct); and on tasks without such labels only (Complete; possibly a selected subset).', star=True)
NQ = json.load(open('paper/data/review_n2_quick.json'))
tabwrap('tab_live_quick', '@{}lrrrrrrc@{}', 'Setting & $n$ & Ref. & Live (loss, $p$) & Replay (loss) & Tok.\\ live / replay & Stop $L_0$/$L_1$/$L_2$ live (replay) & Cert.',
        [f"{nm(k)} & {r['n']} & {r['ref_acc']:.1f} & {r['live_acc']:.1f} (${r['live_loss']:+.1f}$, {r['p']:.2f}) & {r['replay_acc']:.1f} (${r['replay_loss']:+.1f}$) & "
         f"{r['live_tok']:.0f} / {r['replay_tok']:.0f} & " + '/'.join(f"{v:.0f}" for v in r['live_stop']) + ' (' + '/'.join(f"{v:.0f}" for v in r['replay_stop']) + ') & '
         + ('yes' if r['certified_now'] else 'no') + ' \\\\' for k, r in NQ.items()],
        'Earlier live runs of \\climb{} on the test halves, re-scored with the corrected scorers. Their thresholds were certified '
        'under the original labels; Cert.\\ says whether each would still be certified at $\\eps = 5\\%$ under the corrected labels. '
        'Loss: reference minus \\climb{} accuracy on the same tasks; $p$: exact McNemar.', star=True)
print('wrote review tables')
R6 = RV('r6')['summary']; CF = [('0.2/0.2', '20/20'), ('0.4/0.2', '40/20'), ('0.6/0.2', '60/20'), ('0.4/0.1', '40/10'), ('0.4/0.3', '40/30')]
def r6cell(v, m):
    t = f"{v[m]['tok']:.0f}"
    if m == 'A': return t
    lo, hi = v[m]['diff_ci']
    return t + ('$^{\\downarrow}$' if hi < 0 else '$^{\\uparrow}$' if lo > 0 else '')
body = []
for i, (k, lab) in enumerate(CF):
    for j, b in enumerate(('MATH', 'GAIA', 'FRAMES', 'MuSiQue')):
        v = R6[k][b]
        body.append((f"\\multirow{{4}}{{*}}{{{lab}}}" if j == 0 else '') + f" & {b} & {v['n'][0]}/{v['n'][1]} & " + ' & '.join(r6cell(v, m) for m in 'ABCDE')
                    + f" & {v['A']['direct']:.0f} \\\\" + ('\n\\midrule' if j == 3 and i < len(CF) - 1 else ''))
tabwrap('tab_r6', '@{}llrcccccr@{}', 'Train/cal. (\\%) & Bench. & $n$ & A & B & C & D & E & $\\varnothing$ (A)', body,
        'Escalation policies of \\S\\ref{sec:variants} as the training and calibration shares vary separately (GLM, '
        '$\\eps = 5\\%$, 200 splits each; evaluation on the rest). Tokens (\\% of the reference); $^{\\downarrow}$/$^{\\uparrow}$: '
        'significantly cheaper/more expensive than A (95\\% paired bootstrap interval over splits excludes zero); $n$: training/'
        'calibration tasks; $\\varnothing$: share of splits in which A falls back to the reference. All violation rates $\\le$ '
        + f"{max(R6[k][b][m]['viol'] for k, _ in CF for b in R6[k] for m in 'ABCDE'):.1f}" + '\\%.', star=True)
print('wrote tab_r6')
