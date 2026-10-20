"""Open-book versions of FRAMES and MuSiQue (OPEN_BOOK=1).

MuSiQue: the twenty paragraphs that come with each question (supporting and distracting, in the dataset's
order), as in the standard reading-comprehension setting.
FRAMES: the Wikipedia articles each question links to (wiki_links), fetched by analysis/frames_fetch.py into
data/frames_wiki. When the articles of a question exceed FRAMES_DOC_TOKENS tokens in total, each is cut to a
share of that cap (water-filling: short articles stay whole, the rest share what is left), keeping its head,
where the lead section and the infobox are.

The documents are placed before the question inside task['question'], so every agent of every rung sees the
same context and pays for it in its prompt tokens."""
import hashlib, importlib.util, json, os

import tiktoken

ENC = tiktoken.get_encoding('o200k_base')
WIKI = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'frames_wiki')
CAP = int(os.getenv('FRAMES_DOC_TOKENS', '80000'))
_spec = importlib.util.spec_from_file_location('frames_fetch', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis', 'frames_fetch.py'))
_ff = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ff)
link_titles = _ff.link_titles
HEAD = 'The documents below may help answer the question that follows them.\n\n'


def _compose(docs, q):
    body = '\n\n'.join(f'[Document {i + 1}: {t}]\n{x}' for i, (t, x) in enumerate(docs))
    return f'{HEAD}{body}\n\nQuestion: {q}'


def _fit(docs, cap):
    toks = [ENC.encode(x) for _, x in docs]
    if sum(map(len, toks)) <= cap:
        return docs, False
    left, todo, share = cap, sorted(range(len(docs)), key=lambda i: len(toks[i])), {}
    for k, i in enumerate(todo):
        s = left // (len(todo) - k)
        share[i] = min(len(toks[i]), s); left -= share[i]
    return [(t, ENC.decode(toks[i][:share[i]]) + ('\n[...]' if share[i] < len(toks[i]) else ''))
            for i, (t, _) in enumerate(docs)], True


def musique(tasks):
    from datasets import load_dataset
    P = {f"musique_{it['id']}": it['paragraphs'] for it in load_dataset('dgslibisey/MuSiQue', split='validation')}
    for t in tasks:
        t['question'] = _compose([(p['title'], p['paragraph_text']) for p in P[t['id']]], t['question'])
    return tasks


def frames(tasks):
    from datasets import load_dataset
    L = {f"frames_{it['Unnamed: 0']}": eval(it['wiki_links']) for it in load_dataset('google/frames-benchmark', split='test')}
    cut = missing = 0
    for t in tasks:
        docs = []
        titles = link_titles(L[t['id']])
        for x in titles:
            f = os.path.join(WIKI, hashlib.sha1(x.encode()).hexdigest() + '.json')
            d = json.load(open(f)) if os.path.exists(f) else {}
            if d.get('text'):
                docs.append((d.get('resolved') or x, d['text']))
            else:
                missing += 1
        docs, c = _fit(docs, CAP); cut += c
        t['question'] = _compose(docs, t['question'])
    print(f'  FRAMES open book: {len(tasks)} tasks, {cut} cut to {CAP} tokens, {missing} articles missing', flush=True)
    return tasks
