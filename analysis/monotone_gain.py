"""Check the assumption behind the threshold-optimality proposition: the expected gain from
escalating past a rung, P(pipeline right) - P(rung right), is non-increasing in the rung's
confidence score (mean log-prob). Items are binned into quintiles of the score per cell; for
each bin we report the gain and test monotonicity with Spearman's rho between bin index and gain
(a negative rho supports the assumption)."""
import json, statistics as st, sys
from scipy.stats import spearmanr
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items
out = []
print(f"{'cell':<18}{'rung':<5}" + ''.join(f'{"Q"+str(i+1):>8}' for i in range(5)) + "   rho")
for fam, cells in CELLS.items():
    for ds, ae, nf in cells:
        it = items(ae, nf)
        for rung, lp, ok in (('L0', 'lp0', 'ok0'), ('L1', 'lp1', 'ok1')):
            xs = sorted([x for x in it if x[lp] is not None], key=lambda x: x[lp])
            q = [xs[int(len(xs) * k / 5): int(len(xs) * (k + 1) / 5)] for k in range(5)]
            g = [st.mean(int(x['okP']) - int(x[ok]) for x in b) for b in q]
            rho = spearmanr(range(5), g).correlation
            out.append(dict(fam=fam, ds=ds, rung=rung, gain=g, rho=rho))
            print(f"{fam+' '+ds:<18}{rung:<5}" + ''.join(f'{100*v:>+8.1f}' for v in g) + f"   {rho:+.2f}")
neg = sum(1 for r in out if r['rho'] <= 0); print(f"\nrho <= 0 in {neg} of {len(out)} (cell, rung) pairs")
json.dump(out, open('results/monotone_gain.json', 'w'), indent=1)
