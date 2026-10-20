"""Main results on the final records (three models x MATH, FRAMES, MuSiQue, BBH).
Deployed CLIMB: the depth is chosen on calibration between (a) L0 -> reference, thresholded on the L0
log-probability, and (b) L0 + L1 -> reference, stopping at L1 on the score 10 * agree(L0, L1) + S1, which
orders the tasks whose two cheap answers agree first; each family is certified at delta / 2.
Writes data/main_results.json with, per setting, the pipeline, the CLIMB replay at each tolerance
(accuracy, token share, violation), and the descriptive statistics of the big table: rung accuracies,
rescue/harm of the team against L0 and L1, reproducibility of harms over the 300-task repeat, token costs,
AUROC of the scores, and the agreement rate with L1's accuracy on agreeing and disagreeing tasks.
usage: python analysis/main_results.py records.csv  (MAIN_OUT overrides the output path)"""
import string, json, os, re, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import GRID, HERE
from single_decision import calibrate_multi, two_level, EPS

TR = os.path.join(HERE, '..', '..', 'traces')
TAG = {'gpt-oss-20b': 'gpt-oss-20b', 'gpt-oss-120b': 'gpt-oss-120b', 'Gemma-4-26B-A4B': 'gemma-4-26B-A4B-mk'}
OPEN_BOOK = ('FRAMES', 'MuSiQue')   # run with their documents (OPEN_BOOK=1, _open_book.py)


def trace(model, bench, kind):
    """Path of the run behind a setting's records. kind: 'climb' (forced climb through L0 and L1), 'ref' (the
    reference team), 'repeat' (its 300-task rerun). FRAMES and MuSiQue are the open-book runs, Gemma's served
    locally (standard checkpoint, bf16); MATH and BBH are the closed-book runs, Gemma's through Makora."""
    g = os.getenv('REF_GEN', '5')   # 5: reference reruns with the fixed reviewer verdict; 4: the earlier ones
    if bench == 'BBEH':   # all three models served locally (Gemma at bf16), same tags as the other benchmarks
        t = 'gemma-4-26B-A4B-bf16' if model.startswith('Gemma') else TAG[model]
        tag = {'climb': 'l1r', 'ref': f'gx{g}k', 'repeat': f'gx{g}kr2'}[kind]
    elif bench in OPEN_BOOK:
        t = 'gemma-4-26B-A4B-bf16' if model.startswith('Gemma') else TAG[model]
        tag = {'climb': 'l1rob', 'ref': f'gx{g}kob', 'repeat': f'gx{g}kobr2'}[kind]
    else:
        t = TAG[model]
        tag = {'climb': 'climbx' if bench == 'BBH' and model.startswith('gpt-oss') else 'l1r',   # gpt-oss BBH climb used the new L1
               'ref': f'gx{g}k', 'repeat': f'gx{g}kr2'}[kind]
    return os.path.join(TR, f'tasks_sweep_{t}_{tag}_{AB[bench]}100.jsonl')
AB = {'MATH': 'ma', 'FRAMES': 'fr', 'MuSiQue': 'mu', **({'BBEH': 'bbeh'} if os.getenv('USE_BBEH', '1') == '1' else {'BBH': 'bbh'})}
SPLITS = int(os.environ.get('SPLITS', 200)); DELTA = 0.1


_UNI = str.maketrans({'\u202f': ' ', '\u00a0': ' ', '\u2009': ' ', '\u2007': ' ', '\u200b': '', '\u2212': '-',
                      '\u2013': '-', '\u2014': '-', '\u2044': '/', '\u2215': '/'})


def _ans(s):
    """The final answer line (any case), normalised as in SQuAD (lower case, Unicode spaces and dashes mapped to ASCII,
    punctuation and articles removed), so that agreement does not hinge on formatting."""
    m = re.findall(r'(?i)\banswer\**\s*[:\uff1a]\**\s*(.+)', s or '')   # 'ANSWER:', 'Answer:', '**Answer:**'
    if not m:
        return None
    a = m[-1].replace('**', '').translate(_UNI).lower()
    a = ''.join(c for c in a if c not in string.punctuation)
    a = ' '.join(re.sub(r'\b(a|an|the)\b', ' ', a).split())
    return a or None


def climb_trace(model, bench):
    return trace(model, bench, 'climb')


def agreement(x, model, bench):
    A = {}
    for l in open(climb_trace(model, bench)):
        r = json.loads(l); ex = [e for e in r['rounds'] if e['role'] == 'executor']
        a0, a1 = _ans(ex[0].get('answer_snapshot')), _ans(ex[1].get('answer_snapshot'))
        A[r['task_id']] = a0 is not None and a0 == a1
    return np.array([A[t] for t in x.task_id])


def families(x, ag):
    YK, CK, C1 = x.YK.to_numpy(int), x.CK.to_numpy(float), x.C1.to_numpy(float)
    score = ag * 10 + np.nan_to_num(x.S1.to_numpy(float), nan=-np.inf)
    grid = np.array(sorted(set(list(10 + np.array(GRID)) + list(GRID)), reverse=True))[:, None]
    stop = score[None, :] > grid
    Y = np.vstack([YK, np.where(stop, x.Y1, YK)]); C = np.vstack([CK, np.where(stop, C1, C1 + CK)])
    return [(Y, C, YK[None, :] - Y, YK, CK), two_level(x)]


def replay(F, n):
    rng = np.random.default_rng(0); out = {}
    for e in EPS:
        acc, tok, v = [], [], 0
        for _ in range(SPLITS):
            p = rng.permutation(n); cal, ev = p[:n // 2], p[n // 2:]
            f, j = calibrate_multi(F, cal, e); Y, C, L, _, CK = F[f]
            acc.append(Y[j, ev].mean()); tok.append(C[j, ev].mean() / CK[ev].mean()); v += L[j].mean() > e
        out[str(e)] = dict(acc=100 * float(np.mean(acc)), tok=100 * float(np.mean(tok)), viol=100 * v / SPLITS)
    return out


def repeat_harm(x, model, bench):
    cfg = 'V5-Unc-Fmt'
    p = trace(model, bench, 'repeat')
    if not os.path.exists(p) or not os.path.getsize(p):   # repeat still running
        return None
    R2 = {}
    for l in open(p):
        r = json.loads(l)
        if r['config'] == cfg:
            R2[r['task_id']] = r
    y = x.set_index('task_id'); ids = [i for i in R2 if i in y.index]
    h1 = (y.loc[ids, 'Y0'] == 1) & (y.loc[ids, 'YK'] == 0)
    h2 = pd.Series([bool(y.loc[i, 'Y0'] == 1 and not R2[i]['success']) for i in ids], index=ids)
    return int((h1 & h2).sum()), int(h1.sum())


if __name__ == '__main__':
    d = pd.read_csv(sys.argv[1]); out = {}
    for m in TAG:
        for b in AB:
            x = d[(d.model == m) & (d.bench == b)].reset_index(drop=True)
            if not len(x):
                continue
            ag = agreement(x, m, b); y0, y1, yk = x.Y0.values, x.Y1.values, x.YK.values
            au = lambda y, s: float(roc_auc_score(y, s.fillna(-9))) if 0 < y.mean() < 1 else None
            r = dict(n=len(x), pipe_acc=100 * yk.mean(), pipe_tok=float(x.CK.mean()),
                     L0=100 * y0.mean(), L1=100 * y1.mean(),
                     rescue_L0=100 * ((yk == 1) & (y0 == 0)).mean(), harm_L0=100 * ((yk == 0) & (y0 == 1)).mean(),
                     rescue_L1=100 * ((yk == 1) & (y1 == 0)).mean(), harm_L1=100 * ((yk == 0) & (y1 == 1)).mean(),
                     union_L1_team=100 * ((yk == 1) | (y1 == 1)).mean(), repeat_harm=repeat_harm(x, m, b),
                     tok_L0=float(x.C0.mean()), tok_L1=float((x.C1 - x.C0).mean()),
                     auroc_S0=au(x.Y0, x.S0), auroc_S1=au(x.Y1, x.S1), agree=100 * ag.mean(),
                     L1_acc_agree=100 * y1[ag].mean(), L1_acc_disagree=100 * y1[~ag].mean() if (~ag).any() else None,
                     climb=replay(families(x, ag), len(x)))
            out[f'{m}|{b}'] = r
            c = r['climb']
            print(f"{m:16s} {b:8s} pipe {r['pipe_acc']:.1f}/{r['pipe_tok']/1000:.1f}k | " +
                  ' | '.join(f"{v['acc']:.1f}/{v['tok']:.0f}%" for v in c.values()) +
                  f" | viol {max(v['viol'] for v in c.values()):.1f}% | L0 {r['L0']:.1f} L1 {r['L1']:.1f} | resc/harm vs L0 {r['rescue_L0']:.1f}/{r['harm_L0']:.1f} | repeat {r['repeat_harm'] or 'running'}", flush=True)
    json.dump(out, open(os.getenv('MAIN_OUT') or os.path.join(HERE, '..', 'data', 'main_results.json'), 'w'), indent=1)
