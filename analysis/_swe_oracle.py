#!/usr/bin/env python
"""SWE-bench Lite, oracle-localization variant, with EXECUTABLE patches.

The earlier SWE runs asked for "the corrected code" as free text and judged it
with a token-F1 proxy. That cannot be fed to the official harness, which needs a
unified diff that `git apply` accepts. This module fixes that without asking a
27B model to write diff syntax, which small models do badly:

  * the model is shown the file path and each hunk's ORIGINAL region (the gold
    patch's context and removed lines, never its added lines), labelled;
  * it returns the corrected region for each hunk in the same labelling;
  * the diff is then constructed mechanically here.

Because the context lines come from the gold patch itself, they match the file
at base_commit exactly, so no repository checkout is needed to build the patch.
Localization is given, which is a real limitation and must be labelled as such:
these numbers are an oracle-localization variant of SWE-bench Lite and are not
comparable to published agentic resolve rates.
"""
import difflib
import re
from typing import List, Dict, Tuple

HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$')


def parse_gold(patch: str) -> List[Tuple[str, List[Dict]]]:
    """-> [(path, [{'old_start','orig':[lines],'new':[lines]}])], one entry per file.

    Verified contains 71/500 multi-file patches, so a single-file assumption
    would silently drop hunks and produce patches that apply cleanly but do not
    fix the bug, which is worse than failing loudly.
    """
    files, path, hunks, cur = [], None, None, None
    lines = patch.split('\n')
    # A patch ending in a newline yields a trailing '' from split, which is NOT a
    # blank context line. Counting it appended one phantom line to every final
    # hunk, so the '-' side no longer matched the file and `git apply` rejected
    # it. Self-consistency checks cannot catch this: the emitted header and the
    # emitted body were wrong together.
    if lines and lines[-1] == '':
        lines.pop()
    for ln in lines:
        if ln.startswith('--- '):
            continue
        if ln.startswith('+++ '):
            if path is not None and hunks:
                files.append((path, hunks))
            p2 = ln[4:].strip().split('\t')[0]
            path = p2[2:] if p2.startswith('b/') else p2
            hunks, cur = [], None
            continue
        m = HUNK_RE.match(ln)
        if m:
            cur = {'old_start': int(m.group(1)),
                   'declared_old': int(m.group(2)) if m.group(2) is not None else 1,
                   'orig': [], 'new': []}
            if hunks is None:
                hunks = []
            hunks.append(cur)
            continue
        if cur is None:
            continue
        if ln.startswith('-'):
            cur['orig'].append(ln[1:])
        elif ln.startswith('+'):
            cur['new'].append(ln[1:])
        elif ln.startswith(' '):
            cur['orig'].append(ln[1:]); cur['new'].append(ln[1:])
        elif ln == '':
            cur['orig'].append(''); cur['new'].append('')
    if path is not None and hunks:
        files.append((path, hunks))
    return files



def splice_reply(orig: List[str], rep: List[str]) -> List[str]:
    """Map a reply onto the original region.

    The prompt asks for the corrected code of a whole region, and models often
    return only the lines they changed. Substituting a fragment for the region
    deletes everything around it, which is how 20% of one arm's patches left the
    file unparseable. Align the reply to the region instead: find the span the
    reply actually covers and keep the original lines outside it. A reply that
    does reproduce the whole region aligns to the whole span and is unchanged, so
    this only affects the fragment case, and it is applied to every arm alike.
    """
    if not orig or not rep:
        return rep
    a = [l.strip() for l in orig]
    b = [l.strip() for l in rep]
    blocks = [x for x in difflib.SequenceMatcher(None, a, b, autojunk=False)
              .get_matching_blocks() if x.size]
    if not blocks:
        return rep
    lo = blocks[0].a
    hi = blocks[-1].a + blocks[-1].size
    if lo == 0 and hi == len(orig):
        return rep
    return orig[:lo] + rep + orig[hi:]


def build_patch(files: List[Tuple[str, List[Dict]]],
                replacements: List[List[List[str]]]) -> str:
    """Assemble a unified diff from per-hunk replacement regions.

    Each hunk is emitted as its whole original region removed and the
    replacement added. That is a valid unified diff: `git apply` locates the
    hunk by the '-' lines, which are verbatim from the file.
    """
    out = []
    for (path, hunks), reps in zip(files, replacements):
        chunk, drift = [], 0
        for h, rep in zip(hunks, reps):
            old = h['orig']
            if rep is None:
                continue
            rep = splice_reply(old, rep)
            if rep == old:
                continue
            new_start = h['old_start'] + drift
            chunk.append(f"@@ -{h['old_start']},{len(old)} +{new_start},{len(rep)} @@")
            chunk += ['-' + l for l in old]
            chunk += ['+' + l for l in rep]
            drift += len(rep) - len(old)
        if chunk:
            out += [f'--- a/{path}', f'+++ b/{path}'] + chunk
    return '\n'.join(out) + '\n' if out else ''


def render_task(item: Dict, max_issue: int = 3000, max_code: int = 6000) -> str:
    """Prompt showing every region to fix, numbered globally across files.

    Regions are numbered once across all files rather than restarting per file,
    so the reply parser never has to guess which file a REGION belongs to.
    """
    files = parse_gold(item['patch'])
    total = sum(len(h) for _, h in files)
    parts = [f"Repository: {item.get('repo','')}",
             f"Issue:\n{item.get('problem_statement','')[:max_issue]}",
             "\nBelow are the code region(s) to fix. For EACH region, reply with "
             "the corrected code for that region and nothing else, in exactly "
             "this format, preserving indentation:\n"
             "### REGION 1\n```python\n<corrected code>\n```\n"]
    budget = max_code // max(total, 1)
    n = 0
    for path, hunks in files:
        parts.append(f"\nFile: {path}")
        for h in hunks:
            n += 1
            parts.append(f"### REGION {n}\n```python\n"
                         + '\n'.join(h['orig'])[:budget] + "\n```")
    return '\n'.join(parts)


REGION_RE = re.compile(r'###\s*REGION\s*(\d+)\s*\n+```(?:python)?\n(.*?)```', re.S | re.I)


def parse_reply(reply: str, files: List[Tuple[str, List[Dict]]]) -> List[List[List[str]]]:
    """Model reply -> per-file, per-hunk replacement lines.

    A region the model did not answer stays None, and build_patch then leaves
    that hunk untouched rather than inventing one.
    """
    total = sum(len(h) for _, h in files)
    got = {int(i): body for i, body in REGION_RE.findall(reply or '')}
    if not got:
        # A reply that gives the corrected code without the REGION headers is a
        # correct answer in the wrong wrapper, and discarding it measures the
        # parser rather than the model. This mattered: the arm that answers in a
        # single call labelled its blocks on 118 of 500 instances while producing
        # a code block on all 500, so strict parsing threw away most of its work.
        # Map fenced blocks onto hunks in order when the counts agree, and accept
        # a lone block for a lone hunk.
        blocks = re.findall(r'```(?:python)?\n(.*?)```', reply or '', re.S)
        if len(blocks) == total:
            got = {i + 1: b for i, b in enumerate(blocks)}
        elif blocks and total == 1:
            got[1] = blocks[-1]
    out, n = [], 0
    for _, hunks in files:
        per = []
        for _ in hunks:
            n += 1
            per.append(got[n].rstrip('\n').split('\n') if n in got else None)
        out.append(per)
    return out
