"""Merge the parallel GLM ablation parts (results/abl_parts) into results/ablations_glm.json."""
import json, glob, statistics as st
pol, polv, cal, reps_p, reps_c = {}, {}, {}, 0, 0
for f in sorted(glob.glob(__import__('os').environ.get('POL_GLOB', 'results/final_parts/pol_*.json'))):
    d = json.load(open(f)); reps_p += d['reps']
    for b, m in d['policies_raw'].items():
        for k, v in m.items(): pol.setdefault(b, {}).setdefault(k, []).extend(v)
    for b, m in d['policy_viol_counts'].items():
        for k, v in m.items(): polv.setdefault(b, {}).setdefault(k, 0); polv[b][k] += v
for f in sorted(glob.glob('results/final_parts/cal_*.json')):
    d = json.load(open(f)); reps_c += d['reps']
    for b, m in d['calib_raw'].items():
        for k, v in m.items():
            e = cal.setdefault(b, {}).setdefault(k, {'tok': [], 'viol': 0}); e['tok'] += v['tok']; e['viol'] += v['viol']
out = {'policies': {b: {k: round(100 * st.mean(v), 1) for k, v in m.items()} for b, m in pol.items()},
       'policy_violations_eval': {b: {k: round(100 * v / reps_p, 1) for k, v in m.items()} for b, m in polv.items()},
       'calibration_design': {b: {k: [round(100 * st.mean(v['tok'])), round(100 * v['viol'] / reps_c, 1)] for k, v in m.items()} for b, m in cal.items()},
       'reps': {'policies': reps_p, 'calibration': reps_c}}
json.dump(out, open('results/ablations_glm.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
