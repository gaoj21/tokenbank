"""Write sections/tab_calib.tex (Table 2) from data/final_analyses.json: tokens certified at eps = 5% by
each test of the deployed family, per setting. Bold marks the cheapest row of each column."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(HERE, '..', 'data', 'final_analyses.json')))['factorial']
MODELS = (('gpt-oss-20b', '20b'), ('gpt-oss-120b', '120b'), ('Gemma-4-26B-A4B', 'Gemma'))
BENCH = ('MATH', 'FRAMES', 'MuSiQue', 'BBEH')
ROWS = (('signed|hb', 'Hoeffding--Bentkus, signed loss in $[0,1]$'),
        ('signed|hoeffding', 'Hoeffding, signed loss in $[0,1]$'),
        ('clipped|betting', 'Betting, clipped loss'),
        ('clipped|hb', 'Hoeffding--Bentkus, clipped loss'),
        ('clipped|binomial', 'Binomial, clipped loss'),
        ('signed|betting_bonferroni', 'Betting, signed loss, Bonferroni'),
        ('signed|betting', 'Betting, signed loss (ours)'))
cols = [f'{m}|{b}' for m, _ in MODELS for b in BENCH]
best = {c: min(round(F[c]['0.05'][k]) for k, _ in ROWS) for c in cols}
L = [r'\begin{table*}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
     r'\begin{tabular}{@{}lrrrrrrrrrrrr@{}}', r'\toprule',
     r'Test ($\eps{=}5\%$) & ' + ' & '.join(rf'\multicolumn{{4}}{{c}}{{{a}}}' for _, a in MODELS) + r' \\',
     r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}',
     ' & ' + ' & '.join(['M', 'F', 'Q', 'E'] * 3) + r' \\', r'\midrule']
for k, name in ROWS:
    cells = []
    for c in cols:
        v = round(F[c]['0.05'][k]); cells.append(rf'\textbf{{{v}}}' if v == best[c] else str(v))
    if k == 'signed|betting':
        L.append(r'\midrule')
    L.append(name + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule', r'\end{tabular}',
      r'\caption{Tokens (\% of the pipeline'"'"r's) certified at $\eps = 5\%$ by different tests of the same deployed family, over',
      r'100 splits. The signed loss $\ell \in \{-1,0,1\}$ is mapped to $(\ell+1)/2$ with the tolerance mapped to $(\eps+1)/2$ for the',
      r'bounded-range tests, as in \citet{wynn2026corrupted}. The clipped loss $\max(\ell, 0)$ is the harm indicator. All tests',
      r'are fixed-sequence except Bonferroni, which divides $\delta$ by the number of candidates. Bold: cheapest. M/F/Q/E:',
      r'MATH/FRAMES/MuSiQue/BBEH.}', r'\label{tab:calib}', r'\end{table*}']
L = [l.replace(r'\label{tab:calib}', r'\label{tab:calibfull}').replace(r'\begin{table*}[t]', r'\begin{table*}[p]') for l in L]
open(os.path.join(HERE, '..', 'sections', 'tab_calib_full.tex'), 'w').write('\n'.join(L) + '\n')
# compact version for the main text: mean over the four benchmarks per model, and settings where cheapest
import numpy as np
K = (('signed|hb', 'HB, signed in $[0,1]$'),
     ('clipped|betting', 'Betting, clipped'), ('clipped|binomial', 'Binomial, clipped'),
     ('signed|betting_bonferroni', 'Betting, signed, Bonferroni'), ('signed|betting', 'Betting, signed (ours)'))
C = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
     r'Test ($\eps{=}5\%$) & 20b & 120b & Gemma & Best \\', r'\midrule']
MEANS = {k: [round(np.mean([F[f'{m}|{b}']['0.05'][k] for b in BENCH])) for m, _ in MODELS] for k, _ in K}
NBEST = {k: sum(round(F[c]['0.05'][k]) == best[c] for c in cols) for k, _ in K}
bf = lambda s, ok: rf'\textbf{{{s}}}' if ok else s
for k, name in K:
    if k == 'signed|betting':
        C.append(r'\midrule')
    cells = [bf(f'{v}', v == min(MEANS[j][i] for j, _ in K)) for i, v in enumerate(MEANS[k])]
    cells.append(bf(f'{NBEST[k]}/12', NBEST[k] == max(NBEST.values())))
    C.append(name + ' & ' + ' & '.join(cells) + r' \\')
C += [r'\bottomrule', r'\end{tabular}',
      r'\caption{Tokens (\% of the pipeline'"'"r's, mean over the four benchmarks) certified at $\eps = 5\%$ by tests of',
      r'the same deployed family, and the number of settings where each is cheapest (ties count). HB: Hoeffding--Bentkus',
      r'on the signed loss mapped to $[0, 1]$, as in \citet{wynn2026corrupted}. Clipped: the harm indicator',
      r'$\max(\ell, 0)$. Per-setting values, with Hoeffding and clipped HB, in Table~\ref{tab:calibfull}. Bold: best.}',
      r'\label{tab:calib}', r'\end{table}']
open(os.path.join(HERE, '..', 'sections', 'tab_calib.tex'), 'w').write('\n'.join(C) + '\n')
