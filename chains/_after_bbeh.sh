#!/bin/bash
# After BBEH: assemble the final records (MATH and BBEH closed-book, FRAMES and MuSiQue open-book; reference
# runs with the fixed reviewer, gx5k*), then every analysis and Table 1.
cd /home/j0018/Desktop/tokenbank
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "BBEH ALL DONE\|FAILED" logs/_chain_bbeh.log; do sleep 60; done
grep -q "BBEH ALL DONE" logs/_chain_bbeh.log || { log "BBEH chain failed"; exit 1; }
D=paper/data/parts; mkdir -p $D
for m in gpt-oss-20b gpt-oss-120b; do
    RECORDS_OUT=$D/${m}_l1r.csv SUFFIX=100 .venv/bin/python analysis/export_records_new.py ${m}:${m}:l1r:gx5k
    RECORDS_OUT=$D/${m}_ob.csv SUFFIX=100 .venv/bin/python analysis/export_records_new.py ${m}:${m}:l1rob:gx5kob
done
RECORDS_OUT=$D/gemma_mk.csv SUFFIX=100 .venv/bin/python analysis/export_records_new.py gemma-4-26B-A4B-mk:Gemma-4-26B-A4B:l1r:gx5k
RECORDS_OUT=$D/gemma_bf16_l1r.csv SUFFIX=100 .venv/bin/python analysis/export_records_new.py gemma-4-26B-A4B-bf16:Gemma-4-26B-A4B:l1r:gx5k
RECORDS_OUT=$D/gemma_bf16_ob.csv SUFFIX=100 .venv/bin/python analysis/export_records_new.py gemma-4-26B-A4B-bf16:Gemma-4-26B-A4B:l1rob:gx5kob
.venv/bin/python - <<'PY'
import pandas as pd
D = 'paper/data/parts/'; r = lambda f: pd.read_csv(D + f)
P = []
for m in ('gpt-oss-20b', 'gpt-oss-120b'):
    a = r(f'{m}_l1r.csv'); b = r(f'{m}_ob.csv')
    P += [a[a.bench.isin(['MATH', 'BBEH'])], b[b.bench.isin(['FRAMES', 'MuSiQue'])]]
mk = r('gemma_mk.csv'); gl = r('gemma_bf16_l1r.csv'); go = r('gemma_bf16_ob.csv')
P += [mk[mk.bench == 'MATH'], gl[gl.bench == 'BBEH'], go[go.bench.isin(['FRAMES', 'MuSiQue'])]]
d = pd.concat(P).reset_index(drop=True)
d.to_csv('paper/data/records_final.csv', index=False)
print(d.groupby(['model', 'bench']).size().to_dict())
assert len(d.groupby(['model', 'bench'])) == 12
PY
[ $? = 0 ] || { log "MERGE FAILED"; exit 1; }
bash chains/_rerun_analyses_ob.sh
log "AFTER BBEH DONE"
