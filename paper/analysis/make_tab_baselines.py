"""sections/tab_baselines.tex from data/final_analyses.json (baselines) and data/main_results.json (CLIMB):
tokens (% of the pipeline's) and accuracy loss against the pipeline over the twelve settings, per policy."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, '..', 'data', 'final_analyses.json')))['baselines']['per_setting']
M = json.load(open(os.path.join(HERE, '..', 'data', 'main_results.json')))
K = list(B)
rows = []
for key, name in (('always_L0', 'Always $L_0$'), ('always_L1', 'Always $L_1$')):
    rows.append((name, '', [B[k][key][1] for k in K], [B[k]['pipe'] - B[k][key][0] for k in K]))
rows.append(('Threshold, same cost', '', [B[k]['fixed'][1] for k in K], [B[k]['fixed'][0] for k in K]))
for e, lab in (('0.05', '5'), ('0.1', '10')):
    rows.append((f'\\climb{{}}, $\\eps = {lab}\\%$', '\\checkmark', [M[k]['climb'][e]['tok'] for k in K],
                 [M[k]['pipe_acc'] - M[k]['climb'][e]['acc'] for k in K]))
orc = ('Oracle', '', [B[k]['oracle'][1] for k in K], [B[k]['pipe'] - B[k]['oracle'][0] for k in K])
out = ['\\begin{table}[t]', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{3.5pt}',
       '\\begin{tabular}{@{}lcrrr@{}}', '\\toprule',
       ' & & Tokens & \\multicolumn{2}{c}{Loss (points)} \\\\', '\\cmidrule(lr){4-5}',
       'Policy & Cert. & mean & mean & max \\\\', '\\midrule']
for name, c, t, l in rows + [orc]:
    if name == 'Oracle':
        out.append('\\midrule')
    out.append(f'{name} & {c} & {np.mean(t):.0f}\\% & {np.mean(l):.1f}'.replace('-', '$-$') + f' & {max(l):.1f} \\\\')
    print(name, [round(x) for x in t], [round(x, 1) for x in l])
out += ['\\bottomrule', '\\end{tabular}',
        '\\caption{Uncertified rules against \\climb{} over the twelve settings: tokens as a share of the '
        'pipeline\'s and accuracy lost against it (mean and largest). '
        'The threshold on $L_0$\'s confidence is fixed across settings to spend as much as \\climb{} at '
        '$\\eps = 5\\%$. The oracle stops at the cheapest correct rung and is not deployable.}',
        '\\label{tab:baselines}', '\\end{table}']
open(os.path.join(HERE, '..', 'sections', 'tab_baselines.tex'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[10:17]))
