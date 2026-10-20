"""BIG-Bench Extra Hard scoring, following the fuzzy match of the official evaluation
(github.com/google-deepmind/bbeh, evaluate.py): lower-case both sides, drop the space after commas
and markdown bold, keep the first line, strip a trailing period; then accept exact equality, an
option letter with or without parentheses, numerical equality, equality without quotes, a bracketed
form of the other side, or a trailing question mark. The prediction is the text after the last
'ANSWER:' line when there is one, else after the official 'The answer is:' prefix, else the last
non-empty line."""
import re


def extract(pred: str) -> str:
    pred = pred or ''
    m = re.findall(r'ANSWER:\s*(.+)', pred)
    if m:
        return m[-1].strip()
    m = re.findall(r'(?i)the (?:final )?answer is:?\s*(.+)', pred)
    if m:
        return m[-1].strip()
    lines = [l.strip() for l in pred.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ''


def _prep_pred(p: str) -> str:
    p = extract(p).lower().replace(', ', ',').replace('**', '').replace('$', '')
    p = p.split('\n')[0].strip()
    p = re.sub(r'^\\boxed\{(.*)\}$', r'\1', p)
    return p[:-1] if p.endswith('.') else p


def _prep_ref(r: str) -> str:
    return (r or '').strip().lower().replace(', ', ',')


def fuzzy_match(pred: str, ref: str) -> bool:
    if pred == ref:
        return True
    if len(pred) == 3 and pred[0] == '(' and pred[-1] == ')':
        return pred[1] == ref or pred == ref
    if len(ref) == 3 and ref[0] == '(' and ref[-1] == ')':
        return ref[1] == pred
    try:
        if float(pred) == float(ref):
            return True
    except ValueError:
        pass
    if pred.replace("'", '').replace('"', '') == ref.replace("'", '').replace('"', ''):
        return True
    if f'[{ref}]' == pred or f'[{pred}]' == ref:
        return True
    if pred.endswith('?') and pred[:-1] == ref:
        return True
    return False


def judge_bbeh(pred: str, gold: str) -> bool:
    return fuzzy_match(_prep_pred(pred), _prep_ref(gold))
