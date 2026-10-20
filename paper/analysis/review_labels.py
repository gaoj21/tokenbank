"""N1, steps 1, 3 and 4 on data/records.csv.
Impact: per (model, bench, rung) the share of labels the corrected scorers changed (F) and of labels
kept from the strict original scorer because the stored output was cut before the answer (U).
Sensitivity: the main-table replay (200 splits, eps 2/5/10%) under
  current    labels as used in the paper;
  worst      U labels of the reference set to correct (cheap rungs and seeded L2 left wrong): the
             direction that maximises the loss Y_K - Y_pi;
  best       U labels of the cheap rungs and seeded L2 set to correct (reference left wrong);
  complete   tasks with any U label dropped (may be a selected subset).
Both original scorers produced no false positives in our checks, so a U label can only be a false
negative; worst/best bracket the truth for those labels. Writes data/review_labels.json."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import load, outcome_matrix, SETTINGS, HERE
from review_replay import evaluate, sub

def variant(x, how):
    x = x.copy()
    if how == 'worst': x['YK'] = x.YK | x.UK
    if how == 'best':
        for k in '012': x[f'Y{k}'] = x[f'Y{k}'] | x[f'U{k}']
    if how == 'complete': x = x[(x[['U0', 'U1', 'U2', 'UK']].sum(axis=1) == 0)].reset_index(drop=True)
    return x

if __name__ == '__main__':
    d = load(); out = {}
    for m, b in SETTINGS:
        x = sub(d, m, b); o = dict(n=len(x), F={k: 100 * float(x[f'F{k}'].mean()) for k in '012K'},
                                  U={k: 100 * float(x[f'U{k}'].mean()) for k in '012K'},
                                  any_U=100 * float((x[['U0', 'U1', 'U2', 'UK']].sum(axis=1) > 0).mean()))
        for how in ('current', 'worst', 'best', 'complete'):
            v = variant(x, how); Y, C, L = outcome_matrix(v); YK, CK = v.YK.to_numpy(int), v.CK.to_numpy(float)
            o[how] = {str(e): {k: r[k] for k in ('tok', 'loss', 'viol')} for e in (0.02, 0.05, 0.10)
                      for r in [evaluate(Y, C, L, YK, CK, e)]}
            o[how]['n'] = len(v)
        out[f'{m}|{b}'] = o
        print(m, b, 'anyU %.1f' % o['any_U'], ' '.join(f"{h}: " + '/'.join(f"{100*o[h][e]['tok']:.0f}%,{100*o[h][e]['viol']:.1f}" for e in ('0.05', '0.1'))
                                                     for h in ('current', 'worst', 'best', 'complete')))
    json.dump(out, open(os.path.join(HERE, '..', 'data', 'review_labels.json'), 'w'), indent=1)
