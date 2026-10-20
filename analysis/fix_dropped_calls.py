"""Find and replace tasks whose pipeline lost a call. A call is dropped only after every retry failed: five
attempts, each up to three SDK sends of LLM_TIMEOUT (600 s by default), so a task that lost one ran at least
9,000 s. `find` writes the ids of tasks that ran longer than 8,500 s, per trace file; `merge` replaces those
records with the judged reruns.
usage: python analysis/fix_dropped_calls.py find OUTDIR trace.jsonl ...
       python analysis/fix_dropped_calls.py merge trace.jsonl rerun.jsonl"""
import json, os, shutil, sys

if sys.argv[1] == 'find':
    out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    for f in sys.argv[3:]:
        ids = [r['task_id'] for r in map(json.loads, open(f)) if r.get('wall_clock_ms', 0) > 8_500_000]
        name = os.path.basename(f).replace('.jsonl', '.json')
        json.dump(ids, open(os.path.join(out, name), 'w'))
        print(f'{f}: {len(ids)} tasks to rerun', flush=True)
elif sys.argv[1] == 'merge':
    f, g = sys.argv[2], sys.argv[3]
    new = {r['task_id']: r for r in map(json.loads, open(g))}
    rows = [json.loads(l) for l in open(f)]
    os.makedirs('_archive/traces_dropped_calls', exist_ok=True)
    shutil.copy(f, os.path.join('_archive/traces_dropped_calls', os.path.basename(f)))
    n = 0
    with open(f, 'w') as fh:
        for r in rows:
            if r['task_id'] in new:
                r = new[r['task_id']]; n += 1
            fh.write(json.dumps(r) + '\n')
    print(f'{f}: replaced {n} of {len(new)} reruns', flush=True)
