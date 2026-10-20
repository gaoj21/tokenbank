"""BBH scoring with the answer extraction commonly used for BIG-Bench Hard: the text after the last
'ANSWER:' line, with markdown stripped, compared to the target as follows.
  multiple choice "(X)": correct if the answer is the letter, "(X)", or starts with "(X)" / "X)";
  yes/no, True/False, valid/invalid: the first word of the answer;
  word_sorting: words compared after dropping commas;
  dyck_languages: the completion itself, or the given prefix followed by it;
  otherwise the official-style fuzzy match (_bbeh_judge.fuzzy_match), numbers compared numerically."""
import re
from _bbeh_judge import fuzzy_match

WORDS = {'yes', 'no', 'true', 'false', 'valid', 'invalid'}


def _answer(pred):
    m = re.findall(r'ANSWER:\s*(.+)', pred or '')
    a = m[-1] if m else ((pred or '').strip().splitlines() or [''])[-1]
    return a.replace('**', '').replace('`', '').strip().rstrip('.').strip()


def _toks(s):
    return ' '.join(s.replace(',', ' ').split())


def judge_bbh(pred, gold, question=''):
    a, g = _answer(pred), str(gold).strip()
    al, gl = a.lower(), g.lower()
    if re.fullmatch(r'\([a-z]\)', gl):
        letter = gl[1]
        return bool(re.match(rf'^\(?{letter}\)?(?:$|[\s.:,)])', al))
    if gl in WORDS:
        first = re.split(r'[\s,.;:!]+', al.strip())[0] if al.strip() else ''
        return first == gl
    if _toks(al) == _toks(gl):
        return True
    if 'dyck' in question.lower() or 'parentheses are closed properly' in question.lower():
        m = re.search(r'Input:\s*(.+)', question)
        if m and _toks(al) == _toks(m.group(1).strip() + ' ' + g):
            return True
    try:
        if float(al.replace(',', '')) == float(gl):
            return True
    except ValueError:
        pass
    return fuzzy_match(al, gl)
