#!/bin/bash
# Replaces _chain_gemma_mk.sh for the tail: the eight slow Makora runs (FRAMES/MuSiQue reference and their repeats)
# are restarted at 24 workers each; the six new-L1 runs already in flight keep going. When every Makora run has
# finished, merge the shards, rescore MATH and log the line _post_gemma_mk.sh waits for.
cd /home/j0018/Desktop/tokenbank
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/_chain_gemma_mk.log; }
I=results/climb_live; S=/tmp/claude-1001/-home-j0018-Desktop-tokenbank/f14fef7e-adf9-443b-9307-5c07475168b4/scratchpad
B="BUDGETS=10000000 L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason TOP_LOGPROBS=0 MAX_TOKENS=16384 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0"
R=chains/_gemma_mk_run.sh; F="CONFIG_ONLY=V5-Unc-Fmt"
grun() { MAX_WORKERS=24 bash $R "$@" $B >> logs/_chain_gemma_mk.log & }
for i in 0 1; do grun _gx4ks${i}_fr100 BENCHMARK=FRAMES SPLIT=test $F TASK_IDS_FILE=$I/fr_half$i.json; done
for i in 0 1 2 3; do grun _gx4ks${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 SPLIT=test $F TASK_IDS_FILE=$I/oss_mu_shard$i.json; done
grun _gx4kr2_fr100 BENCHMARK=FRAMES SPLIT=test $F TASK_IDS_FILE=$I/repeat300_fr.json
grun _gx4kr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 SPLIT=test $F TASK_IDS_FILE=$I/repeat300_mu.json
log "restarted 8 runs at 24 workers"
sleep 60
alive() { for p in $(ps -eo pid,args | awk '$2 ~ /python/ && $0 ~ /_budget_sweep_v2/ {print $1}'); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^MODEL_TAG=gemma-4-26B-A4B-mk" && return 0; done; return 1; }
while alive; do sleep 60; done
G=traces/tasks_sweep_gemma-4-26B-A4B-mk
for t in l1r gx4k; do
    cat ${G}_${t}s{0,1}_fr100.jsonl > ${G}_${t}_fr100.jsonl
    cat ${G}_${t}s{0,1,2,3}_mu100.jsonl > ${G}_${t}_mu100.jsonl
    cat ${G}_${t}s{0,1,2,3}_bbh100.jsonl > ${G}_${t}_bbh100.jsonl
done
wc -l ${G}_{l1r,gx4k}_{ma,fr,mu,bbh}100.jsonl ${G}_gx4kr2_{ma,fr,mu,bbh}100.jsonl | tee -a logs/_chain_gemma_mk.log
sed "s#glob.glob('traces/tasks_sweep_\*.jsonl')#glob.glob('traces/tasks_sweep_gemma-4-26B-A4B-mk_*_ma100.jsonl')#" analysis/rescore_math.py > $S/rescore_gemma_mk.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python $S/rescore_gemma_mk.py | tail -1
log "GEMMA MK RUNS DONE (FRAMES/MuSiQue still to be judged)"
