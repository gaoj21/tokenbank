"""sections/tab_ablations_main.tex from data/ablations.json: the compact main-text version of tab:ablations,
mean tokens (% of the reference's) over gpt-oss's eight settings, Gemma's four and all twelve, at eps = 5% and 10%,
with the largest violation rate over all settings and the tolerances 2, 5 and 10%."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, '..', 'data', 'ablations.json')))
BENCH = ['MATH', 'FRAMES', 'MuSiQue', 'BBEH']
GROUPS = [['gpt-oss-20b', 'gpt-oss-120b'], ['Gemma-4-26B-A4B'], ['gpt-oss-20b', 'gpt-oss-120b', 'Gemma-4-26B-A4B']]
ROWS = [('ref_draft', 'Ref.: first draft'), ('ref_L1', 'Ref.: single agent $L_1$'),
        ('random', 'Random score'), ('agree', 'Agreement only'), ('logprob', 'Log-prob.\\ only'),
        ('logprob+agree', 'Log-prob.\\ + agr.\\ (ours)')]
mean = lambda k, e, ms: sum(D[f'{m}|{b}'][k][e]['tok'] for m in ms for b in BENCH) / (4 * len(ms))
out = ['\\begin{table}[t]', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{2.3pt}',
       '\\begin{tabular}{@{}lrrrrrrr@{}}', '\\toprule',
       ' & \\multicolumn{3}{c}{$\\eps = 5\\%$} & \\multicolumn{3}{c}{$\\eps = 10\\%$} & \\\\',
       '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
       'Variant & gpt & Gem. & All & gpt & Gem. & All & Viol. \\\\', '\\midrule']
SCORES = ('random', 'agree', 'logprob', 'logprob+agree')   # bold: cheapest score per column
VAL = {k: [round(mean(k, e, g)) for e in ('0.05', '0.1') for g in GROUPS] for k, _ in ROWS}
for k, name in ROWS:
    v = [f'{x}' for x in VAL[k]]   # no bold here; Table 8 marks the cheapest score per setting
    viol = max(D[f'{m}|{b}'][k][x]['viol'] for m in GROUPS[2] for b in BENCH for x in D[f'{m}|{b}'][k])
    out.append(name + ' & ' + ' & '.join(v) + f' & {viol:.0f}\\% \\\\')
    if k == 'ref_L1':
        out.append('\\midrule')
out += ['\\bottomrule', '\\end{tabular}',
        '\\caption{Ablations: mean tokens as a share of the reference\'s over gpt-oss\'s eight settings, Gemma\'s '
        'four and all twelve. The first two rows change the reference, with the deployed score. The rest change the score '
        'of the deployed ladder (random: mean of five draws), ending with \\climb{}\'s. Viol.: largest violation rate over all '
        'settings and $\\eps \\in \\{2, 5, 10\\}\\%$. Per setting in Table~\\ref{tab:ablations}.}',
        '\\label{tab:ablmain}', '\\end{table}']
open(os.path.join(HERE, '..', 'sections', 'tab_ablations_main.tex'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
