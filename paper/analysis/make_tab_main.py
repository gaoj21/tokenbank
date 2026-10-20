"""Write sections/tab_main.tex from data/main_results.json (Table 1)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, '..', 'data', 'main_results.json')))
NAME = {'gpt-oss-20b': 'gpt-oss-20b', 'gpt-oss-120b': 'gpt-oss-120b', 'Gemma-4-26B-A4B': 'Gemma-4-26B-A4B-it'}
B = ('MATH', 'FRAMES', 'MuSiQue', 'BBEH')
cell = lambda a, t, pa: (f"\\textbf{{{a:.1f}}}" if round(a, 1) >= round(pa, 1) else f"{a:.1f}") + " / " + (f"\\textbf{{{t:.0f}\\%}}" if t < 50 else f"{t:.0f}\\%")
L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc@{}}", r"\toprule",
     r"& & & \multicolumn{3}{c}{\climb{}} & \\", r"\cmidrule(lr){4-6}",
     r"Model & Benchmark & Pipeline & $\eps = 2\%$ & $\eps = 5\%$ & $\eps = 10\%$ & Viol. \\", r"\midrule"]
for i, m in enumerate(NAME):
    for k, b in enumerate(B):
        lead = f"\\multirow{{4}}{{*}}{{{NAME[m]}}}" if k == 0 else ""
        if f'{m}|{b}' not in R:   # still running
            L.append(f"{lead} & {b} & \\multicolumn{{5}}{{c}}{{\\emph{{running}}}} \\\\"); continue
        r = R[f'{m}|{b}']; c = r['climb']; pa = r['pipe_acc']
        L.append(f"{lead} & {b} & {pa:.1f} / {r['pipe_tok']/1000:.1f}k & " + " & ".join(cell(c[e]['acc'], c[e]['tok'], pa) for e in ('0.02', '0.05', '0.1'))
                 + f" & {max(v['viol'] for v in c.values()):.1f}\\% \\\\")
    if i < len(NAME) - 1:
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular*}",
      r"\caption{Replay results: accuracy / tokens as a share of the pipeline's (pipeline: accuracy / mean tokens), averaged over",
      r"200 calibration/evaluation splits. \climb{} stops at $L_0$ on its log-probability or, depending on the depth chosen on the calibration half, at $L_1$ on agreement with $L_0$ and",
      r"its log-probability. Viol.\ is the largest share of splits, over the three",
      r"tolerances, in which the chosen policy's loss on the full recorded pool exceeds $\eps$. Bold: under 50\% of the pipeline's",
      r"tokens, or at least as accurate as the pipeline.}", r"\label{tab:main}", r"\end{table*}"]
open(os.path.join(HERE, '..', 'sections', 'tab_main.tex'), 'w').write('\n'.join(L) + '\n')
