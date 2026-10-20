"""Ablations on the final records (data/records_final.csv), replayed exactly as the main results
(200 calibration/evaluation splits, delta = 0.1, families at delta / 2 when two are certified).

signal     CLIMB with the score replaced, ladder and test unchanged: log-probability and agreement
           (deployed), log-probability only (no agreement), agreement only (no log-probability), and a
           random score independent of everything (mean over 5 draws). The guarantee does not depend on
           the score; the savings do.
reference  the reference CLIMB certifies against: the full team (deployed); the team's first draft
           (planner then executor, no review), whose correctness is the judged label of the first
           executor answer and whose cost is the planner's tokens plus the executor's divided by the
           number of rounds (per-round tokens are not recorded, so this cost is an approximation); and
           the single agent L1, with the ladder L0 -> L1.
Writes data/ablations.json.  usage: python analysis/ablations.py"""
import json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import GRID, HERE
from single_decision import calibrate_multi
from main_results import agreement, trace, TAG, AB

EPS = (0.02, 0.05, 0.10); SPLITS = int(os.environ.get('SPLITS', 200)); DELTA = 0.1
GRID2 = np.array(sorted(set(list(10 + np.array(GRID)) + list(GRID)), reverse=True))[:, None]
G1 = np.array(GRID)[:, None]


def fam_l0(S0, Y0, C0, YK, CK):
    """L0 -> reference, stop when S0 > t."""
    stop = S0[None, :] > G1
    Y = np.vstack([YK, np.where(stop, Y0, YK)]); C = np.vstack([CK, np.where(stop, C0, C0 + CK)])
    return Y, C, YK[None, :] - Y, YK, CK


def fam_l1(S1, Y1, C1, YK, CK, grid=GRID2):
    """L0 + L1 -> reference, stop at L1 when S1 > t."""
    stop = S1[None, :] > grid
    Y = np.vstack([YK, np.where(stop, Y1, YK)]); C = np.vstack([CK, np.where(stop, C1, C1 + CK)])
    return Y, C, YK[None, :] - Y, YK, CK


def replay(F, n):
    rng = np.random.default_rng(0); out = {}
    for e in EPS:
        acc, tok, v = [], [], 0
        for _ in range(SPLITS):
            p = rng.permutation(n); cal, ev = p[:n // 2], p[n // 2:]
            f, j = calibrate_multi(F, cal, e, DELTA); Y, C, L, _, CK = F[f]
            acc.append(Y[j, ev].mean()); tok.append(C[j, ev].mean() / CK[ev].mean()); v += L[j].mean() > e
        out[str(e)] = dict(acc=100 * float(np.mean(acc)), tok=100 * float(np.mean(tok)), viol=100 * v / SPLITS)
    return out


def first_draft(x, m, b):
    """Correctness and approximate cost of the team's first draft, per task of x."""
    R = {}
    for l in open(trace(m, b, 'ref')):
        r = json.loads(l)
        if r.get('config') != 'V5-Unc-Fmt':
            continue
        ex = [e for e in r['rounds'] if e['role'] == 'executor']
        rt = r.get('role_tokens') or {}
        cost = rt.get('planner', 0) + rt.get('executor', 0) / max(len(ex), 1)
        R[r['task_id']] = (int(bool(ex and ex[0].get('is_correct'))), float(cost))
    y = np.array([R.get(t, (0, np.nan))[0] for t in x.task_id]); c = np.array([R.get(t, (0, np.nan))[1] for t in x.task_id])
    return y, c


if __name__ == '__main__':
    d = pd.read_csv(os.path.join(HERE, '..', 'data', 'records_final.csv')); out = {}
    for m in TAG:
        for b in AB:
            x = d[(d.model == m) & (d.bench == b)].reset_index(drop=True)
            if not len(x):
                continue
            n = len(x); ag = agreement(x, m, b).astype(float)
            Y0, Y1, YK = (x[k].to_numpy(int) for k in ('Y0', 'Y1', 'YK'))
            C0, C1, CK = (x[k].to_numpy(float) for k in ('C0', 'C1', 'CK'))
            S0 = np.nan_to_num(x.S0.to_numpy(float), nan=-np.inf); s1 = np.nan_to_num(x.S1.to_numpy(float), nan=-np.inf)
            r = {}
            # signal ablation, reference = team
            r['logprob+agree'] = replay([fam_l1(10 * ag + s1, Y1, C1, YK, CK), fam_l0(S0, Y0, C0, YK, CK)], n)
            r['logprob'] = replay([fam_l1(s1, Y1, C1, YK, CK, G1), fam_l0(S0, Y0, C0, YK, CK)], n)
            r['agree'] = replay([fam_l1(10 * ag - 0.5, Y1, C1, YK, CK)], n)
            rs = []
            for k in range(5):
                g = np.random.default_rng(100 + k)
                rs.append(replay([fam_l1(-g.random(n), Y1, C1, YK, CK, G1), fam_l0(-g.random(n), Y0, C0, YK, CK)], n))
            r['random'] = {e: {k: float(np.mean([q[e][k] for q in rs])) for k in rs[0][e]} for e in rs[0]}
            # reference ablation, deployed score
            yd, cd = first_draft(x, m, b); ok = ~np.isnan(cd)
            xd = dict(Y0=Y0[ok], Y1=Y1[ok], C0=C0[ok], C1=C1[ok], S0=S0[ok], S1=(10 * ag + s1)[ok])
            r['ref_draft'] = replay([fam_l1(xd['S1'], xd['Y1'], xd['C1'], yd[ok], cd[ok]),
                                     fam_l0(xd['S0'], xd['Y0'], xd['C0'], yd[ok], cd[ok])], int(ok.sum()))
            r['ref_draft_acc'] = 100 * float(yd[ok].mean()); r['ref_draft_tok_share'] = 100 * float(cd[ok].mean() / CK[ok].mean())
            r['ref_L1'] = replay([fam_l0(S0, Y0, C0, Y1, C1 - C0)], n)
            r['ref_L1_acc'] = 100 * float(Y1.mean()); r['team_acc'] = 100 * float(YK.mean())
            out[f'{m}|{b}'] = r
            f = lambda k: ' '.join(f"{r[k][e]['tok']:.0f}%/{r[k][e]['viol']:.0f}" for e in r[k])
            print(f"{m:16s} {b:8s} | lp+ag {f('logprob+agree')} | lp {f('logprob')} | ag {f('agree')} | rnd {f('random')} "
                  f"| draft({r['ref_draft_acc']:.1f}) {f('ref_draft')} | L1({r['ref_L1_acc']:.1f}) {f('ref_L1')}", flush=True)
    json.dump(out, open(os.path.join(HERE, '..', 'data', 'ablations.json'), 'w'), indent=1)
