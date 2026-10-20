"""N2 (full, GLM): thresholds certified at eps = 5% on the calibration halves under the corrected
labels (results/climb_live/n2_plan.json); live CLIMB re-run on the untouched test halves, paired
with a fresh run of the reference pipeline, three independent rounds on the same serving stack.
GAIA and FRAMES certify the empty action, where CLIMB is the reference by construction.
Per round: paired loss (fresh reference minus live CLIMB, exact McNemar, task-bootstrap 95% CI;
this interval covers task sampling within a round), token share, stopping frequencies, estimated
calls and wall-clock time; across rounds: their spread (covers generation randomness). The replay
prediction for the same threshold and tasks comes from data/records.csv.
Writes paper/data/review_n2_full.json."""
import json, sys, numpy as np, collections
from scipy.stats import binomtest
sys.path.insert(0, 'paper/analysis')
from climb_core import load, TAU
PLAN = json.load(open('results/climb_live/n2_plan.json')); d = load(); rng = np.random.default_rng(0)
EX = lambda r: [x for x in r['rounds'] if x['role'] == 'executor']
ld = lambda f: {json.loads(l)['task_id']: json.loads(l) for l in open(f)}
out = {}
for b, ab in (('MATH', 'ma'), ('MuSiQue', 'mu')):
    t = PLAN[b]['threshold']; ids = json.load(open(f'results/climb_live/n2_test_ids_GLM_{b}.json'))
    x = d[(d.model == 'GLM') & (d.bench == b)].set_index('task_id').loc[ids]
    s0 = (x.S0 > t) & (x.H0 <= TAU); s1 = ~s0 & (x.S1 > t) & (x.H1 <= TAU)
    yr = np.where(s0, x.Y0, np.where(s1, x.Y1, x.Y2)); cr = np.where(s0, x.C0, np.where(s1, x.C1, x.C2))
    rep = dict(loss=100 * (x.YK.mean() - yr.mean()), tok=100 * cr.mean() / x.CK.mean(),
               stop=[100 * s0.mean(), 100 * s1.mean(), 100 * (1 - s0.mean() - s1.mean())], ref_acc=100 * x.YK.mean())
    rounds = []; refs = []
    for r in (1, 2, 3):
        C = ld(f'traces/tasks_sweep_GLM-5.3-Flash_n2c{r}_{ab}24.jsonl'); P = ld(f'traces/tasks_sweep_GLM-5.3-Flash_n2p{r}_{ab}24.jsonl')
        T = [i for i in ids if i in C and i in P]
        yc = np.array([int(C[i]['success']) for i in T]); yp = np.array([int(P[i]['success']) for i in T])
        cc = np.array([C[i]['total_used'] for i in T], float); cp = np.array([P[i]['total_used'] for i in T], float)
        lv = collections.Counter(C[i]['escalation_level'] for i in T)
        calls_c = np.mean([1 if C[i]['escalation_level'] == 0 else 2 if C[i]['escalation_level'] == 1 else 3 + 2 * (len(EX(C[i])) - 2) for i in T])
        calls_p = np.mean([1 + 2 * len(EX(P[i])) for i in T])
        sec_c = np.array([C[i].get('wall_clock_ms', 0) / 1000 for i in T]); sec_p = np.array([P[i].get('wall_clock_ms', 0) / 1000 for i in T])
        L = yp - yc; bs = [L[rng.integers(0, len(L), len(L))].mean() for _ in range(2000)]
        b01 = int(((yp == 1) & (yc == 0)).sum()); b10 = int(((yp == 0) & (yc == 1)).sum())
        rounds.append(dict(n=len(T), ref_acc=100 * yp.mean(), climb_acc=100 * yc.mean(), loss=100 * L.mean(),
                           loss_ci=[100 * np.quantile(bs, .025), 100 * np.quantile(bs, .975)],
                           p=binomtest(b10, b01 + b10).pvalue if b01 + b10 else 1.0, tok=100 * cc.mean() / cp.mean(),
                           stop=[100 * lv[k] / len(T) for k in (0, 1, 2)], calls=[float(calls_c), float(calls_p)],
                           sec=[float(sec_c.mean()), float(sec_p.mean()), float(np.median(sec_c)), float(np.median(sec_p)),
                                float(np.quantile(sec_c, .95)), float(np.quantile(sec_p, .95))]))
        refs.append(yp)
    agree = np.mean([(refs[a] == refs[c]).mean() for a, c in ((0, 1), (0, 2), (1, 2))])
    out[b] = dict(threshold=t, replay=rep, rounds=rounds, ref_agreement=100 * agree,
                  mean=dict(loss=float(np.mean([r['loss'] for r in rounds])), tok=float(np.mean([r['tok'] for r in rounds]))))
    print(b, 't', t, 'replay loss %.1f tok %.0f stop %s' % (rep['loss'], rep['tok'], [round(v) for v in rep['stop']]), 'ref agreement across rounds %.1f%%' % (100 * agree))
    for k, r in enumerate(rounds, 1):
        print(f"  round {k}: n={r['n']} ref {r['ref_acc']:.1f} climb {r['climb_acc']:.1f} loss {r['loss']:+.1f} [{r['loss_ci'][0]:+.1f},{r['loss_ci'][1]:+.1f}] p={r['p']:.2f} tok {r['tok']:.0f}% "
              f"stop {[round(v) for v in r['stop']]} calls {r['calls'][0]:.1f}/{r['calls'][1]:.1f} sec mean {r['sec'][0]:.0f}/{r['sec'][1]:.0f} p95 {r['sec'][4]:.0f}/{r['sec'][5]:.0f}")
json.dump(out, open('paper/data/review_n2_full.json', 'w'), indent=1)
