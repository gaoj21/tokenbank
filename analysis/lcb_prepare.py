"""Prepare LiveCodeBench (code_generation_lite, release_v6 = test.jsonl .. test6.jsonl) for the sweep.
Writes data/lcb/tasks.jsonl (one line per problem: id, prompt fields, test type, function name) and
data/lcb/tests/<id>.json (public + private tests), so a sweep process loads the large hidden test sets
only when it judges a reply. Private tests are stored either as JSON or as base64(zlib(pickle(JSON))),
as in the official loader.
usage: python analysis/lcb_prepare.py"""
import base64, json, os, pickle, zlib

D = 'data/lcb'
os.makedirs(f'{D}/tests', exist_ok=True)


def private(s):
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode('utf-8')))))


n = 0; kinds = {}
with open(f'{D}/tasks.jsonl', 'w') as out:
    for f in ('test.jsonl', 'test2.jsonl', 'test3.jsonl', 'test4.jsonl', 'test5.jsonl', 'test6.jsonl'):
        for line in open(f'{D}/{f}'):
            it = json.loads(line)
            tid = f"lcb_{it['question_id']}"
            tests = json.loads(it['public_test_cases']) + private(it['private_test_cases'])
            meta = json.loads(it.get('metadata') or '{}')
            ttype = tests[0]['testtype'] if tests else 'stdin'
            json.dump([{'input': t['input'], 'output': t['output']} for t in tests], open(f'{D}/tests/{tid}.json', 'w'))
            out.write(json.dumps(dict(id=tid, title=it['question_title'], content=it['question_content'],
                                      starter=it.get('starter_code') or '', platform=it['platform'],
                                      difficulty=it['difficulty'], date=it['contest_date'], testtype=ttype,
                                      func_name=meta.get('func_name'), n_tests=len(tests),
                                      public=json.loads(it['public_test_cases']))) + '\n')
            n += 1; kinds[(it['platform'], it['difficulty'])] = kinds.get((it['platform'], it['difficulty']), 0) + 1
print(n, 'problems'); print(sorted(kinds.items()))
