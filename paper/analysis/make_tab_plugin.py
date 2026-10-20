"""sections/tab_plugin.tex from data/plugin_baseline.json: violation rate and tokens of a per-setting threshold
tuned on the calibration half without certification, against CLIMB, over the twelve settings."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, '..', 'data', 'plugin_baseline.json')))
out = ['\\begin{table}[!t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{4pt}',
       '\\begin{tabular}{@{}lrrrrrr@{}}', '\\toprule',
       ' & \\multicolumn{4}{c}{Violation rate (\\%)} & \\multicolumn{2}{c}{Tokens (\\%)} \\\\',
       '\\cmidrule(lr){2-5}\\cmidrule(l){6-7}',
       ' & \\multicolumn{3}{c}{Tuned} & \\climb{} & & \\\\', '\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}',
       '$\\eps$ & mean & max & $>\\delta$ & max & Tuned & \\climb{} \\\\', '\\midrule']
for e, lab in (('0.02', '2\\%'), ('0.05', '5\\%'), ('0.1', '10\\%')):
    v = lambda k, f: [P[s][e][k][f] for s in P]
    out.append(f"{lab} & {np.mean(v('plugin','viol')):.1f} & {max(v('plugin','viol')):.1f} & "
               f"{sum(x > 10 for x in v('plugin','viol'))}/12 & {max(v('climb','viol')):.1f} & "
               f"{np.mean(v('plugin','tok')):.0f} & {np.mean(v('climb','tok')):.0f} \\\\")
out += ['\\bottomrule', '\\end{tabular}',
        '\\caption{A threshold tuned per setting on the calibration half without certification (Tuned), against '
        '\\climb{}, on the 200 splits of Table~\\ref{tab:main}. Violation rate, as in Table~\\ref{tab:main}: share of '
        'splits in which the deployed policy\'s loss on the full recorded pool exceeds $\\eps$, mean and maximum over the '
        'twelve settings. $>\\delta$: settings above $\\delta = 10\\%$. '
        'Tokens: mean share of the pipeline\'s.}',
        '\\label{tab:plugin}', '\\end{table}']
open(os.path.join(HERE, '..', 'sections', 'tab_plugin.tex'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
