"""Appendix numbers on the final records. Writes data/appendix_extras.json with
  transfer   the policy certified (eps = 5%, full pool of one benchmark) applied to another benchmark of the
             same model: its accuracy loss there;
  delta      tokens at eps = 5% for delta in {0.05, 0.1, 0.2} (100 splits);
  seeding    gpt-oss-20b on MATH: the reference (fresh) against a pipeline seeded with L1's answer and asked to
             verify it (SEED_PROMPT=verify), overall and on the tasks L1 gets wrong;
  labels     answers whose stored reply was cut by the earlier 2,000-character snapshot limit (they keep their
             in-run label), per model and benchmark;
  apierr     tasks in the final records that lost a call after every retry: the closed-book runs' logged
             failures, and for the open-book runs the tasks that ran past the 9,000 s a dropped call takes
             (those of gpt-oss-120b were rerun, analysis/fix_dropped_calls.py).
usage: python analysis/appendix_extras.py"""
import glob, json, os, re, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import HERE
from single_decision import calibrate_multi
from main_results import TAG, AB, families, trace
from final_analyses import frame, SET, evaluate

ROOT = os.path.join(HERE, '..', '..'); TR = os.path.join(ROOT, 'traces')


def transfer():
    out = {}
    for m in TAG:
        bs = [b for b in AB if (m, b) in SET]   # benchmarks with records
        X = {b: frame(m, b) for b in bs}; F = {b: families(X[b], X[b].ag.values) for b in bs}
        for src in bs:
            f, j = calibrate_multi(F[src], np.arange(len(X[src])), 0.05)
            for dst in bs:
                if dst == src:
                    continue
                Y, C, L, _, CK = F[dst][f]
                # the same family and candidate index: thresholds are shared grids, so index j is the same threshold
                out[f'{m}|{src}->{dst}'] = dict(loss=100 * float(L[j].mean()), tok=100 * float(C[j].mean() / CK.mean()))
    return out


def delta_sens(reps=100):
    out = {}
    for m, b in SET:
        x = frame(m, b); F = families(x, x.ag.values); N = len(x); rng = np.random.default_rng(5); r = {}
        for d in (0.05, 0.10, 0.20):
            tok = []
            for _ in range(reps):
                p = rng.permutation(N); tok.append(evaluate(F, p[:N // 2], p[N // 2:], 0.05, d)[1])
            r[str(d)] = 100 * float(np.mean(tok))
        out[f'{m}|{b}'] = r
    return out


def seeding():
    out = {}
    L = lambda p: {json.loads(l)['task_id']: json.loads(l) for l in open(p)}
    for b, seed, ref in (('MATH', 'seedv_ma100', 'gx4k_ma100'),):   # the FRAMES/MuSiQue seeding runs were closed-book
        S = L(os.path.join(TR, f'tasks_sweep_gpt-oss-20b_{seed}.jsonl')); R = L(os.path.join(TR, f'tasks_sweep_gpt-oss-20b_{ref}.jsonl'))
        ids = sorted(set(S) & set(R)); ex = lambda r: [e for e in r['rounds'] if e['role'] == 'executor']
        y1 = np.array([bool(ex(S[i])[1]['is_correct']) for i in ids]); ys = np.array([bool(S[i]['success']) for i in ids])
        yk = np.array([bool(R[i]['success']) for i in ids]); w = ~y1
        out[b] = dict(n=len(ids), fresh=100 * yk.mean(), seeded=100 * ys.mean(), n_L1_wrong=int(w.sum()),
                      fresh_on_L1_wrong=100 * yk[w].mean(), seeded_on_L1_wrong=100 * ys[w].mean())
    return out


def labels():
    out = {}
    for m, b in SET:
        n = cut = 0
        for kind in ('climb', 'ref', 'repeat'):
            if not os.path.exists(trace(m, b, kind)):
                continue
            for l in open(trace(m, b, kind)):
                r = json.loads(l)
                for e in r['rounds']:
                    s = e.get('answer_snapshot') or ''; n += 1
                    cut += len(s) == 2000 and 'ANSWER:' not in s and '\\boxed' not in s
        out[f'{m}|{b}'] = dict(answers=n, cut=cut, share=100 * cut / max(n, 1))
    return out


def apierr():
    out = {}
    for m in TAG:
        n = 0
        for b in AB:
            if (m, b) not in SET:
                continue
            if b in ('FRAMES', 'MuSiQue'):
                for kind in ('climb', 'ref', 'repeat'):
                    if os.path.exists(trace(m, b, kind)):
                        n += sum(1 for l in open(trace(m, b, kind)) if json.loads(l).get('wall_clock_ms', 0) > 8_500_000)
                continue
            for f in glob.glob(os.path.join(ROOT, 'logs', f'{TAG[m]}_*_{AB[b]}100.log')):
                if re.search(r'_(l1r|gx4k|gx4kr2)', f) or (b == 'BBH' and '_climbx' in f):
                    n += sum(1 for l in open(f, errors='ignore') if '[APIERR]' in l)
        out[m] = n
    return out


if __name__ == '__main__':
    res = {}
    for name, fn in (('labels', labels), ('apierr', apierr), ('seeding', seeding), ('transfer', transfer), ('delta', delta_sens)):
        res[name] = fn(); print('done', name, flush=True)
        json.dump(res, open(os.path.join(HERE, '..', 'data', 'appendix_extras.json'), 'w'), indent=1)
