#!/usr/bin/env python
"""Did the constructor change help, and did it leave everything else alone?

Splits each arm by whether splice_reply altered its patch. The unchanged half is
a control: those patches are byte-identical between the two evaluation runs, so
any difference in their outcome is evaluation noise rather than the fix, and a
large difference would mean the comparison itself is unreliable.
"""
import json, os, math, collections

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); ln2 = n * math.log(2)
    ls = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-ln2 for i in range(k+1)]
    m = max(ls)
    return min(1.0, 2*math.exp(m + math.log(sum(math.exp(l-m) for l in ls))))

def rep(rid):
    out = {}
    for root, _, fs in os.walk(f'logs/run_evaluation/{rid}'):
        for f in fs:
            if f == 'report.json':
                for k, v in json.load(open(os.path.join(root, f))).items():
                    out[k] = bool(v.get('resolved'))
    return out

ARMS = [('naive', 'swe_preds_naive.json', 'swe_preds_naive_spliced.json', 'fin_naive', 'sp_naive'),
        ('ours',  'swe_preds_ours.json',  'swe_preds_ours_spliced.json',  'fin_ours',  'sp_ours'),
        ('ab7',   'swe_preds_ab7fix.json','swe_preds_ab7fix_spliced.json','fin_ab7',   'sp_ab7')]

print(f'{"arm":6}{"subset":12}{"n":>6}{"before":>9}{"after":>8}{"gain":>7}{"w/l":>9}{"p":>9}')
print('-' * 66)
for name, oldf, newf, oldrid, newrid in ARMS:
    if not os.path.exists(newf): continue
    old = {x['instance_id']: x['model_patch'] for x in json.load(open(oldf))}
    new = {x['instance_id']: x['model_patch'] for x in json.load(open(newf))}
    RO, RN = rep(oldrid), rep(newrid)
    if not RN: 
        print(f'{name:6}{"(pending)":12}')
        continue
    for lab, sel in (('unchanged', lambda i: old.get(i) == new.get(i)),
                     ('spliced',   lambda i: old.get(i) != new.get(i))):
        ids = sorted(i for i in new if sel(i) and i in RO and i in RN)
        if not ids: continue
        b = sum(RO[i] for i in ids); a = sum(RN[i] for i in ids)
        cw = sum(1 for i in ids if RO[i] and not RN[i])
        wc = sum(1 for i in ids if not RO[i] and RN[i])
        print(f'{name:6}{lab:12}{len(ids):6}{b:9}{a:8}{a-b:+7}{f"{wc}/{cw}":>9}{mcnemar(cw,wc):9.3f}')
    print()
