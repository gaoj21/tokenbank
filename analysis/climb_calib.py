"""CLIMB calibration (final version). A threshold t is certified if the hypothesis
H_t: E[loss] > eps is rejected by either of two valid tests at level delta/2 (union bound):
  (a) exact binomial test on the harm indicator H = 1[pipeline right, policy wrong] >= loss;
  (b) betting test on the signed loss l = Y_pipeline - Y_policy in {-1, 0, 1}
      (capital prod(1 + lam_i (eps - l_i)), predictable lam_i; p = 1 / max capital, Ville).
Thresholds are tested in a fixed order from conservative to aggressive, ending at -inf (always
stop at the cheap rungs); testing stops at the first non-rejection; the cheapest certified
setting on the calibration set is deployed ('direct' = always run the pipeline)."""
import math, statistics as st
from scipy.stats import binom
from risk_gate import apply, GRID
GRIDX = GRID + [round(-0.6 - 0.05 * i, 2) for i in range(1, 49)] + [-1e9]

def p_binom(h, eps):
    return binom.cdf(sum(h), len(h), eps)

def p_betting(l, eps, delta):
    n = len(l); K, Kmax, s1, s2 = 1.0, 1.0, 0.0, 0.0
    cap = 0.5 / (1 - eps)
    for t, x in enumerate(l, 1):
        var = max((s2 + 0.25) / t - (s1 / t) ** 2, 1e-3)          # uses earlier samples only
        lam = min(cap, math.sqrt(2 * math.log(1 / delta) / (var * n)))
        K *= 1 + lam * (eps - x); Kmax = max(Kmax, K)
        s1 += x; s2 += x * x
    return 1.0 / Kmax

def certified(cal, lam, eps, delta, test=None, policy=apply):
    test = test or DEFAULT_TEST
    r = [policy(x, lam) for x in cal]
    h = [int(x['okP'] and not ok) for x, (ok, c) in zip(cal, r)]
    l = [int(x['okP']) - int(ok) for x, (ok, c) in zip(cal, r)]
    if test == 'binom': return p_binom(h, eps) <= delta
    if test == 'betting': return p_betting(l, eps, delta) <= delta
    return p_binom(h, eps) <= delta / 2 or p_betting(l, eps, delta / 2) <= delta / 2

DEFAULT_TEST = 'betting'   # final method: betting test on the signed loss (see results/ablations_glm.json)
def calibrate(cal, eps, delta=0.1, grid=None, test=None, policy=apply):
    test = test or DEFAULT_TEST
    grid = grid or (GRIDX if test != 'binom' else GRID)
    cert = ['direct']
    for lam in grid[1:]:
        if certified(cal, lam, eps, delta, test, policy): cert.append(lam)
        else: break
    return min(cert, key=lambda lam: st.mean(policy(x, lam)[1] for x in cal))
