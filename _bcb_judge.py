#!/usr/bin/env python
"""Execution judge for BigCodeBench, sandboxed in the official evaluation image.

Why a container rather than a subprocess: the benchmark spans 139 libraries
(pandas, sklearn, matplotlib, scipy ...) that this project's venv does not have,
and the code being run is model-generated, so it should not touch the host
filesystem or network. The official image ships both the dependency set and the
isolation.

Why one long-lived container rather than one per call: judging runs inline
during the sweep, once per round, so roughly 24k executions. Container startup
would dominate; `docker exec` into a running container costs ~0.2s.

The container is started on first use and reused. Each call writes a script into
a bind-mounted directory and executes it with a wall-clock timeout; a task that
hangs (BigCodeBench has tasks that can block on input or sleep) is a failure,
not a stall of the whole sweep.
"""
import os, re, subprocess, tempfile, threading, uuid

IMAGE = 'bigcodebench/bigcodebench-evaluate:latest'
NAME = 'bcb-judge'
HOST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bcb_work')
CONT_DIR = '/bcbwork'
TIMEOUT = int(os.getenv('BCB_TIMEOUT', '90'))
_lock = threading.Lock()
_started = False


_NEED_SG = None


def _docker(args, **kw):
    """Run docker, falling back to `sg docker -c` when this process lacks the
    docker group.

    The group was granted to the account after these worker processes' shells
    started, so they inherit an environment without it and every docker call
    fails with EACCES on the socket. That surfaces as a NON-ZERO EXIT with a
    permission message, not FileNotFoundError, so a fallback that only catches
    the missing-binary case never fires and every judgement silently returns
    False. Probe once, then remember which form works.
    """
    global _NEED_SG
    kw.setdefault('capture_output', True)
    kw.setdefault('text', True)
    if _NEED_SG is None:
        try:
            probe = subprocess.run(['docker', 'version', '--format', '{{.Server.Version}}'],
                                   capture_output=True, text=True, timeout=30)
            _NEED_SG = probe.returncode != 0
        except Exception:
            _NEED_SG = True
    if _NEED_SG:
        import shlex
        cmd = 'docker ' + ' '.join(shlex.quote(a) for a in args)
        return subprocess.run(['sg', 'docker', '-c', cmd], **kw)
    return subprocess.run(['docker'] + args, **kw)


def _ensure_container():
    global _started
    with _lock:
        if _started:
            return
        os.makedirs(HOST_DIR, exist_ok=True)
        r = _docker(['inspect', '-f', '{{.State.Running}}', NAME],
                    capture_output=True, text=True)
        if r.returncode != 0 or 'true' not in (r.stdout or ''):
            _docker(['rm', '-f', NAME], capture_output=True)
            # Sized for the whole sweep, not for one test. The first version
            # gave this container 2 CPUs while 64 sweep workers issued
            # concurrent `docker exec` judgements into it; tests then hit the
            # 30s wall clock through contention alone and were recorded as
            # wrong. That is invisible in the output: a starved judge and a
            # failing model both produce success=False.
            _docker(['run', '-d', '--name', NAME, '--network', 'none',
                     '--memory', '32g', '--cpus', str(max(8, (os.cpu_count() or 8) // 2)),
                     '-v', f'{HOST_DIR}:{CONT_DIR}', '--entrypoint', 'sleep',
                     IMAGE, 'infinity'], capture_output=True, text=True)
        _started = True


CODE_BLOCK = re.compile(r'```(?:python)?\s*\n(.*?)```', re.S)


def extract_code(reply: str) -> str:
    """Pull the solution out of a chatty reply.

    Prefer the last fenced block that defines task_func, since models often show
    a wrong draft first and the corrected version last. Fall back to the raw text
    when the model answered without fences.
    """
    blocks = CODE_BLOCK.findall(reply or '')
    withfn = [b for b in blocks if 'def task_func' in b]
    if withfn:
        return withfn[-1]
    if blocks:
        return blocks[-1]
    return reply or ''


def judge_bigcodebench(reply: str, test_code: str) -> bool:
    """True when the generated task_func passes the task's own unittest suite."""
    sol = extract_code(reply)
    if 'def task_func' not in sol:
        return False
    _ensure_container()
    name = f'{uuid.uuid4().hex}.py'
    script = (sol + '\n\n' + test_code + '\n\n'
              "import unittest, sys\n"
              "_r = unittest.main(module=__name__, exit=False, argv=['x'],\n"
              "                   testRunner=unittest.TextTestRunner(verbosity=0)).result\n"
              "sys.exit(0 if _r.wasSuccessful() else 1)\n")
    path = os.path.join(HOST_DIR, name)
    try:
        with open(path, 'w') as fh:
            fh.write(script)
        r = _docker(['exec', NAME, 'timeout', str(TIMEOUT),
                     'python', f'{CONT_DIR}/{name}'],
                    capture_output=True, text=True, timeout=TIMEOUT + 15)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
