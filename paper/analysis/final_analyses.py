"""Every number of the paper beyond the main table, on the final records (data/records_final.csv), for the
deployed procedure of main_results.py. Writes data/final_analyses.json with
  baselines   uncertified rules on the full pool: always L0, always L1, the fixed L0 threshold that spends as
              much as CLIMB at eps = 5% (mean over settings), the oracle; and threshold transfer across models;
  sweep       tokens and violations of CLIMB as the tolerance varies (100 splits per point);
  sizes       tokens as the calibration set grows (eps = 5%, 10%);
  variants    escalation designs certified the same way (train 20 / calibrate 40 / evaluate 40):
              A L0 alone, B shared log-prob threshold on L0 and L1, C agreement at L1, D = deployed (A or C at
              delta/2), E learned signed gain (gradient-boosted classifier on the training split);
  factorial   loss (signed / clipped) x test (betting / Hoeffding-Bentkus / Hoeffding / binomial), plus betting
              with a Bonferroni correction instead of fixed-sequence testing;
  monotone    gain of escalating (Y_ref - Y_0) by L0-confidence quintile, its Spearman correlation, pooled deciles;
  team        what the team adds: cost against L1, coordination share, rescue/harm against L1, reviewer
              acceptance of right and wrong answers, union with L1;
  calib_cost  tokens to record the calibration set and the queries needed to repay it.
usage: python analysis/final_analyses.py"""
import collections, json, os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climb_core import GRID, HERE, calibrate, p_betting
from single_decision import two_level, ladder_ref, calibrate_multi
from main_results import TAG, AB, agreement, families, trace
from loss_test_factorial import p_hb, p_hoeffding, p_binom

DELTA = 0.1; TR = os.path.join(HERE, '..', '..', 'traces')
D = pd.read_csv(os.path.join(HERE, '..', 'data', 'records_final.csv'))
SET = [(m, b) for m in TAG for b in AB if ((D.model == m) & (D.bench == b)).any()]   # settings with records


def frame(m, b):
    x = D[(D.model == m) & (D.bench == b)].reset_index(drop=True)
    x['ag'] = agreement(x, m, b)
    return x


def evaluate(F, cal, ev, e, delta=DELTA):
    f, j = calibrate_multi(F, cal, e, delta)
    Y, C, L, _, CK = F[f]
    return Y[j, ev].mean(), C[j, ev].mean() / CK[ev].mean(), L[j].mean() > e


def l0_policy(x, t):
    S0 = x.S0.fillna(-9).values; s = S0 > t
    y = np.where(s, x.Y0, x.YK); c = np.where(s, x.C0, x.C0 + x.CK)
    return 100 * (x.YK.mean() - y.mean()), 100 * c.mean() / x.CK.mean()


def baselines(X, main):
    out = {}
    for (m, b), x in X.items():
        CK = x.CK.mean()
        orc_y = np.where(x.Y0 == 1, 1, np.where(x.Y1 == 1, 1, x.YK))
        orc_c = np.where(x.Y0 == 1, x.C0, np.where(x.Y1 == 1, x.C1, x.C1 + x.CK))
        out[f'{m}|{b}'] = dict(pipe=100 * x.YK.mean(), always_L0=(100 * x.Y0.mean(), 100 * x.C0.mean() / CK),
                               always_L1=(100 * x.Y1.mean(), 100 * x.C1.mean() / CK),
                               oracle=(100 * orc_y.mean(), 100 * orc_c.mean() / CK))
    target = np.mean([main[k]['climb']['0.05']['tok'] for k in main])
    t = min(GRID, key=lambda t: abs(np.mean([l0_policy(x, t)[1] for x in X.values()]) - target))
    for (m, b), x in X.items():
        out[f'{m}|{b}']['fixed'] = l0_policy(x, t)
    tr = {}
    for src in TAG:
        S = [x for (mm, _), x in X.items() if mm == src]
        tb = min([t for t in GRID if all(l0_policy(x, t)[0] <= 5 for x in S)], key=lambda t: np.mean([l0_policy(x, t)[1] for x in S]))
        tr[src] = dict(t=tb, to={dst: max(l0_policy(x, tb)[0] for (mm, _), x in X.items() if mm == dst) for dst in TAG if dst != src})
    return dict(per_setting=out, fixed_t=t, target_tok=target, transfer=tr)


def sweep(X, eps=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15), reps=100):
    out = {}
    for (m, b), x in X.items():
        F = families(x, x.ag.values); N = len(x); rng = np.random.default_rng(1); r = {}
        for e in eps:
            tok, v = [], 0
            for _ in range(reps):
                p = rng.permutation(N); a, t, vv = evaluate(F, p[:N // 2], p[N // 2:], e); tok.append(t); v += vv
            r[str(e)] = dict(tok=100 * float(np.mean(tok)), viol=100 * v / reps)
        out[f'{m}|{b}'] = r
    return out


def sizes(X, ns=(50, 100, 200, 400, 800), reps=100):
    out = {}
    for (m, b), x in X.items():
        F = families(x, x.ag.values); N = len(x); rng = np.random.default_rng(2); r = {}
        for n in ns:
            if n > 0.8 * N:
                continue
            for e in (0.05, 0.10):
                tok = []
                for _ in range(reps):
                    p = rng.permutation(N); tok.append(evaluate(F, p[:n], p[n:], e)[1])
                r[f'{n}|{e}'] = 100 * float(np.mean(tok))
        out[f'{m}|{b}'] = r
    return out


def learned_family(x, tr):
    from sklearn.ensemble import GradientBoostingClassifier
    feats = np.column_stack([x.S0.fillna(-9), x.S1.fillna(-9), x.ag.astype(float), x.C0, x.C1 - x.C0])
    gain = (x.YK - x.Y1).to_numpy()                         # signed gain of escalating instead of stopping at L1
    if len(np.unique(gain[tr])) < 2:                      # nothing to learn on this split: fall back to the agreement score
        return families(x, x.ag.values)[0]
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
    clf.fit(feats[tr], gain[tr]); cls = list(clf.classes_); P = clf.predict_proba(feats)
    g = (P[:, cls.index(1)] if 1 in cls else 0) - (P[:, cls.index(-1)] if -1 in cls else 0)
    sc = -g; grid = list(np.unique(np.quantile(sc[tr], np.linspace(0.99, 0.0, 40)))[::-1]) + [-np.inf]
    YK, CK, C1 = x.YK.to_numpy(int), x.CK.to_numpy(float), x.C1.to_numpy(float)
    stop = sc[None, :] > np.array(grid)[:, None]
    Y = np.vstack([YK, np.where(stop, x.Y1, YK)]); C = np.vstack([CK, np.where(stop, C1, C1 + CK)])
    return (Y, C, YK[None, :] - Y, YK, CK)


def variants(X, reps=100):
    out = {}
    for (m, b), x in X.items():
        N = len(x); rng = np.random.default_rng(3); fam = families(x, x.ag.values)
        V = dict(A=[two_level(x)], B=[ladder_ref(x)], C=[fam[0]], D=fam)
        res = {k: {e: [] for e in ('0.05', '0.1')} for k in 'ABCDE'}
        for _ in range(reps):
            p = rng.permutation(N); tr, cal, ev = p[:int(.2 * N)], p[int(.2 * N):int(.6 * N)], p[int(.6 * N):]
            E = [learned_family(x, tr)]
            for k, F in list(V.items()) + [('E', E)]:
                for e in (0.05, 0.10):
                    res[k][str(e)].append(evaluate(F, cal, ev, e, DELTA)[1])
        out[f'{m}|{b}'] = {k: {e: 100 * float(np.mean(v)) for e, v in d.items()} for k, d in res.items()}
    return out


def factorial(X, reps=100):
    cells = [('signed', 'betting'), ('signed', 'hb'), ('signed', 'hoeffding'), ('clipped', 'betting'),
             ('clipped', 'hb'), ('clipped', 'hoeffding'), ('clipped', 'binomial'), ('signed', 'betting_bonferroni')]

    def pv(L, loss, test, e, delta):
        if loss == 'signed':
            if test.startswith('betting'):
                return p_betting(L, e, delta)
            Z, a = (L + 1) / 2, (e + 1) / 2
        else:
            Z, a = np.maximum(L, 0).astype(float), e
            if test == 'betting':
                return p_betting(Z, e, delta)
            if test == 'binomial':
                return p_binom(Z, a)
        return p_hb(Z, a) if test == 'hb' else p_hoeffding(Z, a)

    def select(L, C, loss, test, e, delta):
        p = pv(L[1:], loss, test, e, delta)
        if test == 'betting_bonferroni':                   # every candidate tested at delta / M, no ordering
            ok = np.concatenate([[True], p <= delta / len(p)])
            return int(np.argmin(np.where(ok, C.mean(1), np.inf)))
        fail = np.nonzero(p > delta)[0]; k = fail[0] if len(fail) else len(p)
        return int(np.argmin(C[:k + 1].mean(1)))

    out = {}
    for (m, b), x in X.items():
        F = families(x, x.ag.values); N = len(x); rng = np.random.default_rng(4); r = {}
        for e in (0.05, 0.10):
            acc = {f'{l}|{t}': [] for l, t in cells}
            for _ in range(reps):
                p = rng.permutation(N); cal, ev = p[:N // 2], p[N // 2:]
                for l, t in cells:
                    best = None
                    for Y, C, L, _, CK in F:
                        j = select(L[:, cal], C[:, cal], l, t, e, DELTA / len(F)); c = C[j, cal].mean()
                        if best is None or c < best[0]:
                            best = (c, C[j, ev].mean() / CK[ev].mean())
                    acc[f'{l}|{t}'].append(best[1])
            r[str(e)] = {k: 100 * float(np.mean(v)) for k, v in acc.items()}
        out[f'{m}|{b}'] = r
    return out


def monotone(X):
    out = {}; pooled = []
    for (m, b), x in X.items():
        s = x.S0.fillna(-9).to_numpy(); g = (x.YK - x.Y0).to_numpy(); c = x.CK.to_numpy(float)
        q = pd.qcut(pd.Series(s).rank(method='first'), 5, labels=False).to_numpy()
        gq = [100 * float(g[q == k].mean()) for k in range(5)]; gc = [float(g[q == k].mean() / c[q == k].mean() * 1e4) for k in range(5)]
        out[f'{m}|{b}'] = dict(gain_by_quintile=gq, gain_per_10k_tokens=gc, rho=float(spearmanr(range(5), gq)[0]))
        pooled.append(pd.DataFrame(dict(r=pd.Series(s).rank(pct=True), g=g)))
    P = pd.concat(pooled); P['dec'] = np.minimum((P.r * 10).astype(int), 9)
    return dict(per_setting=out, pooled_deciles=[100 * float(P[P.dec == k].g.mean()) for k in range(10)])


def team(X):
    out = {}
    for (m, b), x in X.items():
        R = [json.loads(l) for l in open(trace(m, b, 'ref'))]
        rt = collections.Counter(); ar = aw = nr = nw = 0; rounds = []
        for r in R:
            for k, v in (r.get('role_tokens') or {}).items():
                rt[k.split('_')[0]] += v
            ex = [e for e in r['rounds'] if e['role'] == 'executor']; rounds.append(len(ex))
            for e in ex:
                if 'review_accept' in e:
                    if e['is_correct']:
                        nr += 1; ar += e['review_accept']
                    else:
                        nw += 1; aw += e['review_accept']
        y1, yk = x.Y1.values, x.YK.values
        out[f'{m}|{b}'] = dict(cost_vs_L1=float(x.CK.mean() / (x.C1 - x.C0).mean()),
                               coordination=100 * (rt['planner'] + rt['reviewer']) / sum(rt.values()),
                               gain_vs_L1=100 * (yk.mean() - y1.mean()), rescue_L1=100 * ((yk == 1) & (y1 == 0)).mean(),
                               harm_L1=100 * ((yk == 0) & (y1 == 1)).mean(), union=100 * ((yk == 1) | (y1 == 1)).mean(),
                               accept_right=100 * ar / max(nr, 1), accept_wrong=100 * aw / max(nw, 1),
                               one_round=100 * float(np.mean(np.array(rounds) == 1)))
    return out


def calib_cost(X, main):
    out = {}
    for (m, b), x in X.items():
        n = len(x) // 2; rec = n * float((x.C1 + x.CK).mean()); CK = float(x.CK.mean()); r = dict(n=n, tokens=rec)
        for e in ('0.05', '0.1'):
            save = CK * (1 - main[f'{m}|{b}']['climb'][e]['tok'] / 100)
            r[e] = rec / save if save > 0 else None
        out[f'{m}|{b}'] = r
    return out


if __name__ == '__main__':
    main = json.load(open(os.path.join(HERE, '..', 'data', 'main_results.json')))
    X = {s: frame(*s) for s in SET}
    path = os.path.join(HERE, '..', 'data', 'final_analyses.json')
    only = os.environ.get('STAGES')
    res = json.load(open(path)) if only and os.path.exists(path) else {}
    for name, fn in (('baselines', lambda: baselines(X, main)), ('monotone', lambda: monotone(X)), ('team', lambda: team(X)),
                     ('calib_cost', lambda: calib_cost(X, main)), ('sweep', lambda: sweep(X)), ('sizes', lambda: sizes(X)),
                     ('factorial', lambda: factorial(X)), ('variants', lambda: variants(X))):
        if only and name not in only.split(','):
            continue
        res[name] = fn(); print('done', name, flush=True)
        json.dump(res, open(path, 'w'), indent=1)
