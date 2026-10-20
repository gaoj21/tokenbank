"""MATH answer checking (fixed). The previous judge passed answers to math-verify without math
delimiters, so plain fractions ("16/27"), degrees ("36°"), unicode ("√3", "204₅") and assignments
("x = 2") were often rejected even when equal to the reference. Here both sides are parsed inside
$...$, with a light normalisation of common non-LaTeX notations, and a string fallback."""
import re
from math_verify import parse, verify

_SUB = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
_SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')

def _extract(text):
    text = text or ''
    boxed = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    cands = [boxed[-1]] if boxed else []
    m = re.findall(r'ANSWER:\s*(.+)', text)
    if m: cands.append(m[-1].strip())
    m = list(re.finditer(r'answer\s*(?:is|are|was|were)?\s*[:\-]\s*(.+)', text, re.I))
    if m: cands.append(m[-1].group(1).strip())
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines: cands.append(lines[-1])
    return cands

_UNI = {'\u202f': ' ', '\u00a0': ' ', '\u2009': ' ', '\u2007': ' ', '\u200b': '', '\u2212': '-', '\u2013': '-',
        '\u2044': '/', '\u2215': '/'}


def _norm(s):
    # Unicode spacing and dashes (gpt-oss writes "x\u202f\u2212\u202f6"): map them to ASCII before parsing.
    for k, v in _UNI.items():
        s = s.replace(k, v)
    s = s.replace('**', '').strip().rstrip('.').strip()          # markdown bold around the answer
    s = re.sub(r'^\\\(|\\\)$', '', s).strip().strip('$').strip()
    s = re.sub(r'√\s*(?:\(([^()]+)\)|([0-9a-zA-Z]+))', lambda m: '\\sqrt{' + (m.group(1) or m.group(2)) + '}', s)
    s = re.sub(r'([0-9A-Za-z]+)([₀₁₂₃₄₅₆₇₈₉]+)', lambda m: m.group(1) + '_' + m.group(2).translate(_SUB), s)
    s = re.sub(r'([⁰¹²³⁴⁵⁶⁷⁸⁹]+)', lambda m: '^' + m.group(1).translate(_SUP), s)
    s = s.replace('π', '\\pi').replace('∞', '\\infty').replace('°', '^\\circ').replace('·', '\\cdot ').replace('×', '\\times ')
    s = re.sub(r'^[a-zA-Z]\s*=\s*(?!.*=)', '', s)          # "x = 2" -> "2" (single assignment only)
    return s

def _parse(s):
    out = []
    for form in ('$' + s + '$', s):
        try:
            p = parse(form)
            if p: out.append(p)
        except Exception:
            pass
    return out

def judge_math(pred, gold):
    gold = str(gold)
    golds = _parse(gold) + _parse(_norm(gold))
    for c in _extract(pred):
        for form in {c, _norm(c)}:
            for p in _parse(form):
                for g in golds:
                    try:
                        if verify(g, p):
                            return True
                    except Exception:
                        pass
    n = lambda s: re.sub(r'[\s\$\\,{}]|\^\\circ|°', '', _norm(str(s))).rstrip('.')
    return any(n(c) == n(gold) for c in _extract(pred))
