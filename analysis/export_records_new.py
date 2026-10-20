"""Export per-task records for the new models (Gemma-4-31B, gpt-oss) in the records.csv format.
For each (model tag, benchmark): the forced-climb run (CLIMB-Live, THETA_LIVE=99) gives L0, L1 and the
seeded L2; the reference run (V5-Unc-Fmt) gives YK, CK. Also counts tasks whose pipeline hit the
budget (closing call). usage: python analysis/export_records_new.py TAG:MODEL_NAME[:climbtag:reftag] ... -> paper/data/records_new.csv"""
import csv, json, sys, os
EX = lambda d: [r for r in d['rounds'] if r['role'] == 'executor']
def load(path, config):
    out = {}
    for l in open(path):
        d = json.loads(l)
        if d['config'] == config: out[d['task_id']] = d
    return out
F = ['model', 'bench', 'task_id', 'S0', 'H0', 'Y0', 'C0', 'S1', 'H1', 'Y1', 'C1', 'Y2', 'C2', 'YK', 'CK', 'budget_ref', 'budget_climb']
rows = []; stats = {}
for spec in sys.argv[1:]:
    tag, name, ctag, rtag = (spec.split(':') + ['climbx', 'gx4k'])[:4]
    for b, ab in (('MATH', 'ma'), ('GAIA', 'ga'), ('FRAMES', 'fr'), ('MuSiQue', 'mu'), ('BigCodeBench', 'bcb'), ('LiveCodeBench', 'lcb'), ('MBPP+', 'mbpp'), ('BBH', 'bbh'), ('BBEH', 'bbeh')):
        sfx = os.environ.get('SUFFIX', '100')                      # 100: no per-task budget (100k safety cap)
        pc, pr = f'traces/tasks_sweep_{tag}_{ctag}_{ab}{sfx}.jsonl', f'traces/tasks_sweep_{tag}_{rtag}_{ab}{sfx}.jsonl'
        if not (os.path.exists(pc) and os.path.exists(pr) and os.path.getsize(pc) and os.path.getsize(pr)): continue
        cc, rc = ('CLIMB-Live-BCB', 'BCB-Naive') if ab in ('bcb', 'lcb', 'mbpp') else ('CLIMB-Live', 'V5-Unc-Fmt')
        cx, ref = load(pc, cc), load(pr, rc); n = drop = 0
        for i in sorted(set(cx) & set(ref)):
            d, r = cx[i], ref[i]; ex = EX(d)
            if len(ex) < 2 or ex[0].get('reserve_after') is None or ex[1].get('reserve_after') is None: drop += 1; continue
            B = d['budget']
            rows.append([name, b, i, ex[0].get('mean_logprob'), ex[0].get('entropy'), int(bool(ex[0]['is_correct'])), B - ex[0]['reserve_after'],
                         ex[1].get('mean_logprob'), ex[1].get('entropy'), int(bool(ex[1]['is_correct'])), B - ex[1]['reserve_after'],
                         int(bool(d['success'])), d['total_used'], int(bool(r['success'])), r['total_used'],
                         int(r.get('stopped_by') == 'budget'), int(d.get('stopped_by') == 'budget')]); n += 1
        stats[f'{name}|{b}'] = dict(n=n, dropped=drop)
os.makedirs('paper/data', exist_ok=True)
with open(os.environ.get('RECORDS_OUT', 'paper/data/records_new.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(F); w.writerows(rows)
print(json.dumps(stats))
