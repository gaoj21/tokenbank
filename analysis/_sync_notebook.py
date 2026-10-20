"""One-off: sync notebook code cells from _run_v2_experiment.py sections.

The .py is the source of truth; sections marked '# ============ NOTEBOOK CELL N ============'
map 1:1 onto code cells of token_bank_v2_experiment.ipynb.
"""
import json
import re

PY = '_run_v2_experiment.py'
NB = 'token_bank_v2_experiment.ipynb'

py_src = open(PY).read()
parts = re.split(r'# ============ NOTEBOOK CELL (\d+) ============\n', py_src)
cell_map = {}
for i in range(1, len(parts), 2):
    cell_map[int(parts[i])] = parts[i + 1].strip('\n')

nb = json.load(open(NB))
updated = []
for idx, content in sorted(cell_map.items()):
    cell = nb['cells'][idx]
    assert cell['cell_type'] == 'code', f'cell {idx} is not code'
    lines = content.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    cell['outputs'] = []
    cell['execution_count'] = None
    updated.append(idx)

json.dump(nb, open(NB, 'w'), indent=1, ensure_ascii=False)
print('updated cells:', updated)
