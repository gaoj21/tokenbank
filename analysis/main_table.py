"""Main replay table with the final calibration (analysis/climb_calib.py): both models, four
benchmarks, eps in {1, 2, 5}%, 200 calibration/evaluation splits; violations measured both on the
evaluation half and against the risk on the full pool. Also baselines on the full test set."""
import json, random, statistics as st, sys
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items, apply, top
from climb_calib import calibrate
REPS = 200
def oracle(x):
    if x['ok0']: return True, x['c0']
    if x['ok1']: return True, x['c1']
    return top(x)
if __name__ == '__main__':
    out = {}
    random.seed(7)
    for fam, cells in CELLS.items():
        for ds, ae, nf in cells:
            it = items(ae, nf); cP = st.mean(x['cP'] for x in it); aP = 100 * st.mean(x['okP'] for x in it)
            pool = {}
            def prisk(lam):
                if lam not in pool: pool[lam] = st.mean(int(x['okP']) - int(apply(x, lam)[0]) for x in it)
                return pool[lam]
            def summ(f): r = [f(x) for x in it]; return [round(100 * st.mean(ok for ok, c in r), 1), round(100 * st.mean(c for ok, c in r) / cP)]
            row = dict(n=len(it), pipe_acc=round(aP, 1), pipe_tok=round(cP), L0=summ(lambda x: (x['ok0'], x['c0'])),
                       L1=summ(lambda x: (x['ok1'], x['c1'])), fixed=summ(lambda x: apply(x, -0.10)), oracle=summ(oracle))
            for eps in tuple(float(e) for e in __import__('os').environ.get('EPS_LIST', '0.01,0.02,0.05').split(',')):
                tok, loss, ve, vp = [], [], 0, 0
                for _ in range(REPS):
                    random.shuffle(it); h = len(it) // 2; cal, te = it[:h], it[h:]
                    lam = calibrate(cal, eps); r = [apply(x, lam) for x in te]
                    l = st.mean(int(x['okP']) - int(ok) for x, (ok, c) in zip(te, r))
                    tok.append(st.mean(c for ok, c in r) / st.mean(x['cP'] for x in te)); loss.append(l)
                    ve += l > eps; vp += prisk(lam) > eps
                row[f'eps{int(100 * eps)}'] = dict(acc=round(aP - 100 * st.mean(loss), 1), tok=round(100 * st.mean(tok)),
                                                   viol_eval=round(100 * ve / REPS, 1), viol_pool=round(100 * vp / REPS, 1))
            out[f'{fam}|{ds}'] = row
            print(fam, ds, {k: v for k, v in row.items() if k.startswith('eps')}, flush=True)
    json.dump(out, open(__import__('os').environ.get('MAIN_OUT', 'results/main_table.json'), 'w'), indent=1)
