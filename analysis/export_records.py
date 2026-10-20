"""Export the per-task records behind every replay number in the paper to paper/data/records.csv.
One row per (model, benchmark, task): confidence S_k (mean token log-prob), mean top-5 entropy H_k,
correctness Y_k and cumulative ledger cost C_k of L0, L1 and the seeded L2 from the deployed-CLIMB
records, and correctness/cost of the reference pipeline (Y_K, C_K) that the loss is measured against.
Label-audit columns, per rung k in {0, 1, 2, K}:
  F_k = 1 if the corrected scorer changed the label from incorrect to correct (MATH: fixed
        math-verify judge; FRAMES/MuSiQue: LLM judge; GAIA: never re-scored);
  U_k = 1 if the label is 0 and could not be re-scored because the stored output (cut at 2,000
        characters) does not contain the final answer, so the strict original label was kept.
Both original scorers had no false positives in our checks, so only labels with U_k = 1 are in doubt.
Also: estimated model calls N_k up to each rung (traces log executor rounds only; the pipeline
adds one planner call and one reviewer call per round; tool follow-ups excluded) and wall-clock seconds of the full climb (T2) and of the
reference run (TK); per-rung timings were not recorded."""
import csv, json, os, re, sys
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items, _climbx_path
cut_math = lambda a: len(a or '') >= 2000 and not re.search(r'ANSWER:|\\boxed', a or '')
cut_em = lambda a: len(a or '') >= 2000 and 'ANSWER:' not in (a or '')
EX = lambda d: [r for r in d['rounds'] if r['role'] == 'executor']

def load(path, config=None):
    out = {}
    for l in open(path):
        d = json.loads(l)
        if config is None or d['config'] == config: out[d['task_id']] = d
    return out

def labels(d):
    """Executor-round labels (L0, L1) and task-level label with the snapshot each was judged from."""
    ex = EX(d)
    return [(bool(ex[0]['is_correct']), ex[0].get('answer_snapshot')), (bool(ex[1]['is_correct']), ex[1].get('answer_snapshot')),
            (bool(d['success']), ex[-1].get('answer_snapshot') if ex else None)] if len(ex) >= 2 else None

F = ['model', 'bench', 'task_id', 'S0', 'H0', 'Y0', 'C0', 'S1', 'H1', 'Y1', 'C1', 'Y2', 'C2', 'YK', 'CK',
     'F0', 'F1', 'F2', 'FK', 'U0', 'U1', 'U2', 'UK', 'N0', 'N1', 'N2', 'NK', 'T2', 'TK']
with open('paper/data/records.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(F); n = 0
    for fam, cells in CELLS.items():
        for ds, ae, nf in cells:
            cut = cut_math if ds == 'MATH' else cut_em
            arch = {'MATH': '_archive/traces_math_oldjudge', 'FRAMES': '_archive/traces_em', 'MuSiQue': '_archive/traces_em'}.get(ds)
            cx, ref = load(_climbx_path(ae)), load(nf, 'V5-Unc-Fmt')
            ocx = load(os.path.join(arch, os.path.basename(_climbx_path(ae)))) if arch else cx
            oref = load(os.path.join(arch, os.path.basename(nf)), 'V5-Unc-Fmt') if arch else ref
            for x in sorted(items(ae, nf, keep_id=True), key=lambda x: x['id']):
                i = x['id']; new = labels(cx[i]); old = labels(ocx[i])
                nK = (bool(ref[i]['success']), EX(ref[i])[-1].get('answer_snapshot') if EX(ref[i]) else None)
                oK = bool(oref[i]['success'])
                Fs = [int(nw[0] and not od[0]) for nw, od in zip(new, old)] + [int(nK[0] and not oK)]
                Us = [int((not lab) and arch is not None and cut(snap)) for lab, snap in new + [nK]]
                w.writerow([fam, ds, i, x['lp0'], x['e0'], int(x['ok0']), x['c0'], x['lp1'], x['e1'], int(x['ok1']), x['c1'],
                            int(x['okS']), x['cS'], int(x['okP']), x['cP'], *Fs, *Us,
                            1, 2, 2 + 1 + 2 * (len(EX(cx[i])) - 2), 1 + 2 * len(EX(ref[i])),
                            round(cx[i].get('wall_clock_ms', 0) / 1000, 1), round(ref[i].get('wall_clock_ms', 0) / 1000, 1)]); n += 1
print('rows', n)
