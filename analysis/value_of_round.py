"""Does one more round of the loop pay? From the full-counterfactual traces at cap 4096, every
item ran the compressed loop for up to five rounds and each round's answer was judged. For
consecutive loop rounds t -> t+1, count wrong->right (repair) and right->wrong (damage), and
relate them to the budget left at round t."""
import json, os, collections, statistics as st
R = lambda d: [r for r in d['rounds'] if r['role'] == 'executor']
C = [('GLM','MATH','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_ma24.jsonl'),('GLM','GAIA','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_ga24.jsonl'),
     ('GLM','FRAMES','traces/tasks_sweep_GLM-5.3-Flash_glmx4k_fr24.jsonl'),('GLM','MuSiQue','traces/tasks_sweep_GLM-5.3-Flash_glmx4ka_mu24.jsonl'),
     ('DeepSeek','MATH','traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_ma24.jsonl'),('DeepSeek','GAIA','traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_ga24.jsonl'),
     ('DeepSeek','FRAMES','traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_fr24.jsonl'),('DeepSeek','MuSiQue','traces/tasks_sweep_DeepSeek-V4.1-Flash_dsx4k_mu24.jsonl')]
tot = collections.Counter(); byrem = collections.defaultdict(collections.Counter); byt = collections.defaultdict(collections.Counter)
print(f"{'':<18}{'transitions':>12}{'repair':>8}{'damage':>8}{'net/100':>9}")
for fam, ds, p in C:
    c = collections.Counter()
    for l in open(p):
        d = json.loads(l)
        if not d['config'].endswith('AllEsc'): continue
        rs = R(d); loop = rs[2:]
        used = sum(r['prompt_tokens'] + r['completion_tokens'] for r in rs[:2])
        for t in range(len(loop) - 1):
            used += loop[t]['prompt_tokens'] + loop[t]['completion_tokens']
            a, b = bool(loop[t]['is_correct']), bool(loop[t+1]['is_correct'])
            k = 'same' if a == b else ('repair' if b else 'damage')
            c[k] += 1; tot[k] += 1
            rem = 24000 - used
            bucket = '>16k' if rem > 16000 else ('8-16k' if rem > 8000 else '<8k')
            byrem[bucket][k] += 1; byt[t+1][k] += 1
    n = sum(c.values())
    print(f"{fam+' '+ds:<18}{n:>12}{c['repair']:>8}{c['damage']:>8}{100*(c['repair']-c['damage'])/max(n,1):>+9.2f}")
n = sum(tot.values())
print(f"{'pooled':<18}{n:>12}{tot['repair']:>8}{tot['damage']:>8}{100*(tot['repair']-tot['damage'])/n:>+9.2f}")
print("\nby budget remaining before the extra round")
for b in ['>16k','8-16k','<8k']:
    c = byrem[b]; n = sum(c.values())
    if n: print(f"  {b:<6} n={n:<6} repair {c['repair']:<5} damage {c['damage']:<5} net/100 {100*(c['repair']-c['damage'])/n:+.2f}")
print("by round transition")
for t in sorted(byt):
    c = byt[t]; n = sum(c.values()); print(f"  round {t}->{t+1}  n={n:<6} repair {c['repair']:<5} damage {c['damage']:<5} net/100 {100*(c['repair']-c['damage'])/n:+.2f}")
