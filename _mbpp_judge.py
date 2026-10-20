"""Execution judge for MBPP+ (EvalPlus), run in the network-less BigCodeBench container (_bcb_judge):
the reply's last ```python block followed by the task's own MBPP+ test script (base and extra inputs
with expected outputs); the reply passes when the script exits cleanly."""
import json, os, re, shutil, uuid
from _bcb_judge import _docker, _ensure_container, HOST_DIR, NAME, CONT_DIR

CODE_BLOCK = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)
TIMEOUT = int(os.getenv('MBPP_TIMEOUT', '60'))
_TESTS = None


def _tests():
    global _TESTS
    if _TESTS is None:
        import pandas as pd
        d = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mbppplus', 'test.parquet'))
        _TESTS = {f"mbpp_{r.task_id}": (r.test_imports, r.test) for r in d.itertuples()}
    return _TESTS


def judge_mbpp(reply: str, task_id: str) -> bool:
    blocks = CODE_BLOCK.findall(reply or '')
    if not blocks:
        return False
    imports, test = _tests()[task_id]
    imp = '\n'.join(json.loads(imports) if isinstance(imports, str) and imports.startswith('[') else list(imports))
    script = imp + '\n' + blocks[-1] + '\n\n' + test + '\n'
    _ensure_container()
    tag = uuid.uuid4().hex; work = os.path.join(HOST_DIR, f'mbpp_{tag}'); os.makedirs(work)
    try:
        open(os.path.join(work, 'run.py'), 'w').write(script)
        r = _docker(['exec', '-w', f'{CONT_DIR}/mbpp_{tag}', NAME, 'timeout', str(TIMEOUT), 'python', 'run.py'],
                    capture_output=True, text=True, timeout=TIMEOUT + 30)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)
