"""sections/tab_ablations.tex from data/ablations.json: tokens (% of the pipeline's, or of the reference's for the
reference rows) at eps = 5% and 10% for each ablation, per setting, with the largest violation rate per row."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, '..', 'data', 'ablations.json')))
MODELS = ['gpt-oss-20b', 'gpt-oss-120b', 'Gemma-4-26B-A4B']; BENCH = ['MATH', 'FRAMES', 'MuSiQue', 'BBEH']
ROWS = [('ref_draft', 'Reference: the team\'s first draft'), ('ref_L1', 'Reference: the single agent $L_1$'),
        ('random', 'Random score'), ('agree', 'Agreement only'), ('logprob', 'Log-prob.\\ only'),
        ('logprob+agree', 'Log-prob.\\ and agreement (\\climb{})')]
out = ['\\begin{table*}[h]', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{2.6pt}',
       '\\begin{tabular}{@{}l' + 'r' * 12 + 'rr@{}}', '\\toprule',
       ' & \\multicolumn{4}{c}{20b} & \\multicolumn{4}{c}{120b} & \\multicolumn{4}{c}{Gemma} & & \\\\',
       '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}',
       'Variant & M & F & Q & E & M & F & Q & E & M & F & Q & E & Mean & Viol. \\\\', '\\midrule']
for e, lab in (('0.05', '5'), ('0.1', '10')):
    out.append(f'\\multicolumn{{15}}{{@{{}}l}}{{\\emph{{$\\eps = {lab}\\%$}}}} \\\\')
    SCORES = ('random', 'agree', 'logprob', 'logprob+agree')   # bold: cheapest score per column
    VAL = {}
    for k, _ in ROWS:
        v = [D[f'{m}|{b}'][k][e]['tok'] for m in MODELS for b in BENCH]
        VAL[k] = [round(x) for x in v] + [round(sum(v) / len(v))]
    for k, name in ROWS:
        viol = max(D[f'{m}|{b}'][k][x]['viol'] for m in MODELS for b in BENCH for x in D[f'{m}|{b}'][k])
        cells = [f'\\textbf{{{x}}}' if k in SCORES and x == min(VAL[j][i] for j in SCORES) else f'{x}' for i, x in enumerate(VAL[k])]
        out.append(name + ' & ' + ' & '.join(cells) + f' & {viol:.1f}\\% \\\\')
        if k == 'ref_L1':
            out.append('\\addlinespace[2pt]')
    out.append('\\midrule' if e == '0.05' else '\\bottomrule')
out += ['\\end{tabular}',
        '\\caption{Ablations of the score and of the reference, replayed on the same 200 splits as Table~\\ref{tab:main}: '
        'tokens as a share of the reference\'s, at $\\eps = 5\\%$ and $10\\%$. The first two rows change the reference that '
        'the loss is measured against, with the deployed score. The other four change only the score of the deployed '
        'ladder (the random score is drawn independently of everything, mean of five draws), ending with \\climb{}\'s. The first draft\'s cost is '
        'approximate (planner tokens plus the executor\'s divided by the number of rounds). Viol.: largest violation rate '
        'over the twelve settings and the three tolerances 2, 5 and 10\\%. Bold: cheapest score. M/F/Q/E: MATH/FRAMES/MuSiQue/BBEH.}',
        '\\label{tab:ablations}', '\\end{table*}']
open(os.path.join(HERE, '..', 'sections', 'tab_ablations.tex'), 'w').write('\n'.join(out) + '\n')
print('wrote sections/tab_ablations.tex')
