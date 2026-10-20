"""Execution judge for LiveCodeBench, run in the same network-less container as the BigCodeBench
judge (_bcb_judge). A reply passes when its last ```python block passes every public and hidden test,
following the official evaluation: stdin problems run the program on each input and compare the
output token by token (numbers within 1e-6); functional (LeetCode) problems call
Solution().<func_name>(*args) with the JSON-decoded input lines and compare the JSON-decoded output.
Each test has a 6 s limit (LCB_TEST_TIMEOUT); the first failure ends the run."""
import json, os, re, shutil, subprocess, uuid
from _bcb_judge import _docker, _ensure_container, HOST_DIR, NAME, CONT_DIR

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'lcb', 'tests')
PER_TEST = float(os.getenv('LCB_TEST_TIMEOUT', '6'))
TOTAL = int(os.getenv('LCB_TIMEOUT', '240'))
PRELUDE = ('from typing import *\nimport math, sys, re, string, heapq, bisect, itertools, functools, collections, random\n'
           'from collections import *\nfrom functools import *\nfrom itertools import *\nfrom heapq import *\n'
           'from bisect import *\nfrom math import *\nsys.setrecursionlimit(1 << 20)\n')
CODE_BLOCK = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)

RUNNER = r'''
import json, subprocess, sys, math
sol, tests, ttype, fn, per = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
tests = json.load(open(tests))
def same_tokens(a, b):
    a, b = a.split(), b.split()
    if len(a) != len(b): return False
    for x, y in zip(a, b):
        if x == y: continue
        try:
            if not math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6): return False
        except ValueError:
            return False
    return True
prog = sol   # for functional problems the host has already appended the call driver
for t in tests:
    try:
        r = subprocess.run([sys.executable, prog], input=t['input'], capture_output=True, text=True, timeout=per)
    except subprocess.TimeoutExpired:
        sys.exit(1)
    if r.returncode != 0: sys.exit(1)
    if ttype == 'functional':
        try:
            got, exp = json.loads(r.stdout.strip().splitlines()[-1]), json.loads(t['output'])
        except Exception:
            sys.exit(1)
        norm = lambda v: [norm(x) for x in v] if isinstance(v, (list, tuple)) else v
        if norm(got) != norm(exp):
            if not (isinstance(got, float) or isinstance(exp, float)) or not math.isclose(float(got), float(exp), rel_tol=1e-6, abs_tol=1e-6):
                sys.exit(1)
    elif not same_tokens(r.stdout, t['output']):
        sys.exit(1)
sys.exit(0)
'''


def extract_code(reply):
    blocks = CODE_BLOCK.findall(reply or '')
    return blocks[-1] if blocks else ''


def judge_lcb(reply: str, spec: str) -> bool:
    """spec: JSON with the task id, test type and function name (the task's 'answer' field)."""
    s = json.loads(spec); sol = extract_code(reply)
    if not sol.strip():
        return False
    if s['testtype'] == 'functional' and 'class Solution' not in sol:
        return False
    _ensure_container()
    tag = uuid.uuid4().hex; work = os.path.join(HOST_DIR, f'lcb_{tag}'); os.makedirs(work)
    try:
        sol = PRELUDE + sol                  # the official evaluation's common imports (typing, collections, ...)
        if s['testtype'] == 'functional':   # the container cannot write here, so the driver is added now
            sol += ("\n\nimport json as _j, sys as _s\n_a = [_j.loads(l) for l in _s.stdin.read().split('\\n') if l.strip()]\n"
                    f"print(_j.dumps(getattr(Solution(), {s['func_name']!r})(*_a)))\n")
        open(os.path.join(work, 'sol.py'), 'w').write(sol)
        open(os.path.join(work, 'run.py'), 'w').write(RUNNER)
        shutil.copy(os.path.join(TEST_DIR, f"{s['id']}.json"), os.path.join(work, 'tests.json'))
        cw = f'{CONT_DIR}/lcb_{tag}'
        r = _docker(['exec', '-w', cw, NAME, 'timeout', str(TOTAL), 'python', 'run.py', 'sol.py', 'tests.json',
                     s['testtype'], s.get('func_name') or '-', str(PER_TEST)],
                    capture_output=True, text=True, timeout=TOTAL + 30)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)
