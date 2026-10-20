"""Risk-controlled escalation gate (Learn-then-Test), evaluated offline.

Policy family, one parameter lam: stop at L0 if its mean log-prob > lam and entropy <= 0.6;
else stop at L1 if the same holds for L1; else escalate to the full pipeline.
Per item: L0/L1 answers, log-probs and token costs come from the all-escalate traces (cap
4096); the full pipeline's outcome and cost come from the plain pipeline run on the same item
(cap 4096). Escalation cost = tokens spent at L0/L1 + the pipeline's. Caveat: in a real ladder
the pipeline would receive the L0/L1 draft; here it runs independently.

Loss of a policy on an item = 1[pipeline correct] - 1[policy correct], in {-1, 0, 1};
risk = mean loss = accuracy lost relative to always running the pipeline.
Learn-then-Test with fixed-sequence testing, from lam = +inf (always escalate, risk 0) down to
more aggressive lam. The test statistic is the harm indicator h = 1[pipeline right, policy
wrong], which upper-bounds the loss, so bounding E[h] bounds the risk; its p-value is the exact
binomial tail P(Bin(n, eps) <= k). (A Hoeffding bound on the loss itself never rejects at these
sample sizes, because it ignores that almost all losses are zero.) Returns the most
aggressive lam whose null "risk > eps" is rejected at level delta, so that
P(test risk > eps) <= delta over calibration draws.
"""
import json, math, os, random, statistics as st
from scipy.stats import binom
TAU = 0.6
GRID = ['direct'] + [round(-0.005 * i, 3) for i in range(1, 121)]
# 'direct' = skip the ladder and run the pipeline (risk 0, cost of the pipeline alone). The
# ladder is engaged only if calibration certifies some lam within eps; so cost never exceeds
# the pipeline's at the most conservative setting.
R = lambda d: [r for r in d['rounds'] if r['role'] == 'executor']
CELLS = {
 'GLM': [('MATH','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_ma24.jsonl','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_ma24.jsonl'),
         ('GAIA','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_ga24.jsonl','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_ga24.jsonl'),
         ('FRAMES','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_fr24.jsonl','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_fr24.jsonl'),
         ('MuSiQue','traces/tasks_sweep_GLM-5.3-Flash_glmx4ka_mu24.jsonl','traces/tasks_sweep_GLM-5.3-Flash_glmfp8n_mu24.jsonl')],   # MuSiQue reference pipeline re-run on FP8
 'DeepSeek': [(ds, f'traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_{t}24.jsonl', f'traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_{t}24.jsonl')
              for t, ds in [('ma','MATH'),('ga','GAIA'),('fr','FRAMES'),('mu','MuSiQue')]]}
# Replay source. 'climbx' (final): records of the deployed CLIMB configuration in which every task
# climbed L0 -> L1 -> L2 seeded with the draft (THETA_LIVE=99), so replay matches deployment;
# cumulative rung costs come from the budget ledger and include tool calls. The reference
# pipeline (V5-Unc-Fmt, no draft) is unchanged. 'allesc' (legacy): L0/L1 from the AllEsc arms.
SOURCE = os.environ.get('REPLAY_SOURCE', 'climbx')
def _climbx_path(ae_path):
    tag = 'GLM-5.3-Flash' if 'GLM' in ae_path else 'DeepSeek-V4.1-Flash'
    ab = ae_path.rsplit('_', 1)[1][:2]
    return f'traces/tasks_sweep_{tag}_climbx_{ab}24.jsonl'
def _items_climbx(ae_path, nf_path, keep_id=False):
    nf = {}
    for l in open(nf_path):
        d = json.loads(l)
        if d['config'] == 'V5-Unc-Fmt': nf[d['task_id']] = d
    out = []
    for l in open(_climbx_path(ae_path)):
        d = json.loads(l)
        if d['task_id'] not in nf: continue
        rs = R(d)
        if len(rs) < 2 or d.get('escalation_level') != 2 or rs[0].get('reserve_after') is None or rs[1].get('reserve_after') is None:
            continue
        B = d['budget']; i = d['task_id']
        out.append(dict(lp0=rs[0].get('mean_logprob'), e0=rs[0].get('entropy'), ok0=bool(rs[0]['is_correct']),
                        lp1=rs[1].get('mean_logprob'), e1=rs[1].get('entropy'), ok1=bool(rs[1]['is_correct']),
                        c0=B - rs[0]['reserve_after'], c1=B - rs[1]['reserve_after'],
                        okS=bool(d['success']), cS=d['total_used'],
                        okP=bool(nf[i]['success']), cP=nf[i]['total_used'], **({'id': i} if keep_id else {})))
    return out
def items(ae_path, nf_path, keep_id=False):
    if SOURCE == 'climbx':
        return _items_climbx(ae_path, nf_path, keep_id)
    ae, nf = {}, {}
    for l in open(ae_path):
        d = json.loads(l)
        if d['config'].endswith('AllEsc'): ae[d['task_id']] = d
    for l in open(nf_path):
        d = json.loads(l)
        if d['config'] == 'V5-Unc-Fmt': nf[d['task_id']] = d
    out = []
    for i in set(ae) & set(nf):
        rs = R(ae[i])
        if len(rs) < 2: continue
        c0 = rs[0]['prompt_tokens'] + rs[0]['completion_tokens']; c1 = c0 + rs[1]['prompt_tokens'] + rs[1]['completion_tokens']
        out.append(dict(lp0=rs[0].get('mean_logprob'), e0=rs[0].get('entropy'), ok0=bool(rs[0]['is_correct']),
                        lp1=rs[1].get('mean_logprob'), e1=rs[1].get('entropy'), ok1=bool(rs[1]['is_correct']),
                        c0=c0, c1=c1, okP=bool(nf[i]['success']), cP=nf[i]['total_used'], **({'id': i} if keep_id else {})))
    return out
def top(x):
    """Outcome and cumulative cost of a task that climbs to the top rung."""
    return (x['okS'], x['cS']) if 'okS' in x else (x['okP'], x['c1'] + x['cP'])
conf = lambda lp, e, lam: lp is not None and lp > lam and (e is None or e <= TAU)
def apply(it, lam):
    if lam == 'direct': return it['okP'], it['cP']
    if conf(it['lp0'], it['e0'], lam): return it['ok0'], it['c0']
    if conf(it['lp1'], it['e1'], lam): return it['ok1'], it['c1']
    return top(it)                                       # seeded pipeline, as deployed
def risk(S, lam): return st.mean(int(x['okP']) - int(apply(x, lam)[0]) for x in S)
def ltt(cal, eps, delta):
    # Fixed-sequence testing yields a set of certified settings; the guarantee holds for any
    # choice within it, so pick the one that is cheapest on the calibration split.
    n = len(cal); certified = ['direct']
    for lam in GRID[1:]:                 # fixed sequence, conservative to aggressive
        k = sum(1 for x in cal if x['okP'] and not apply(x, lam)[0])
        if binom.cdf(k, n, eps) <= delta: certified.append(lam)
        else: break
    return min(certified, key=lambda lam: st.mean(apply(x, lam)[1] for x in cal))
if __name__ == '__main__':
    random.seed(0)
    res = []
    print(f"{'cell':<18}{'eps':>5}{'test risk':>11}{'violations':>12}{'tokens vs pipeline':>20}{'stop early':>12}   fixed gate (-0.10): risk / tokens")
    for fam, cells in CELLS.items():
        for ds, ae, nf in cells:
            it = items(ae, nf)
            Pc = st.mean(x['cP'] for x in it)
            fr = st.mean(int(x['okP']) - int(apply(x, -0.10)[0]) for x in it); ft = st.mean(apply(x, -0.10)[1] for x in it)
            for eps in (0.01, 0.02, 0.05):
                rs, ts, vs, es = [], [], 0, []
                for _ in range(200):
                    random.shuffle(it); h = len(it) // 2; cal, te = it[:h], it[h:]
                    lam = ltt(cal, eps, 0.1)
                    r = risk(te, lam); rs.append(r); vs += r > eps
                    ts.append(st.mean(apply(x, lam)[1] for x in te) / st.mean(x['cP'] for x in te))
                    es.append(0.0 if lam == 'direct' else st.mean(apply(x, lam)[1] < x['c1'] + x['cP'] for x in te))
                row = dict(fam=fam, ds=ds, n=len(it), eps=eps, risk=st.mean(rs), viol=vs/200, tok=st.mean(ts), early=st.mean(es),
                           fixed_risk=fr, fixed_tok=ft/Pc)
                res.append(row)
                print(f"{fam+' '+ds:<18}{100*eps:>4.0f}%{100*row['risk']:>+10.2f}{100*row['viol']:>11.1f}%{100*row['tok']:>19.0f}%{100*row['early']:>11.0f}%   {100*fr:+.1f} / {100*ft/Pc:.0f}%")
    json.dump(res, open('results/risk_gate.json', 'w'), indent=1)
