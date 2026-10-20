#!/bin/bash
# Gemma-4-26B-A4B-it through OpenRouter/Makora (tag gemma-4-26B-A4B-mk), the same 26 runs as the local QAT
# chain (_chain_gemma_full.sh, tag gemma-4-26B-A4B-qat). The two are never merged; whichever finishes first is used.
cd /home/j0018/Desktop/tokenbank
log() { echo "[$(date '+%F %T')] $*"; }
I=results/climb_live; S=/tmp/claude-1001/-home-j0018-Desktop-tokenbank/f14fef7e-adf9-443b-9307-5c07475168b4/scratchpad
B="BUDGETS=10000000 L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason TOP_LOGPROBS=0 MAX_TOKENS=16384 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0"
R=chains/_gemma_mk_run.sh; C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"; F="CONFIG_ONLY=V5-Unc-Fmt"
grun() { MAX_WORKERS=6 bash $R "${@:2}" $B & }   # 26 runs x 6 = 156 concurrent: Makora rate-limits at ~600
for t in l1r gx4k; do
    [ $t = l1r ] && X="$C" || X="$F"
    grun 24 _${t}_ma100 BENCHMARK=MATH MATH_N=500 SPLIT=test $X
    grun 24 _${t}s0_fr100 BENCHMARK=FRAMES SPLIT=test $X TASK_IDS_FILE=$I/fr_half0.json
    grun 24 _${t}s1_fr100 BENCHMARK=FRAMES SPLIT=test $X TASK_IDS_FILE=$I/fr_half1.json
    for i in 0 1 2 3; do grun 24 _${t}s${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 SPLIT=test $X TASK_IDS_FILE=$I/oss_mu_shard$i.json; done
    for i in 0 1 2 3; do grun 24 _${t}s${i}_bbh100 BENCHMARK=BBH SPLIT=full $X TASK_IDS_FILE=$I/bbh_shard$i.json; done
done
grun 24 _gx4kr2_ma100 BENCHMARK=MATH MATH_N=500 SPLIT=test $F TASK_IDS_FILE=$I/repeat300_ma.json
grun 24 _gx4kr2_fr100 BENCHMARK=FRAMES SPLIT=test $F TASK_IDS_FILE=$I/repeat300_fr.json
grun 24 _gx4kr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 SPLIT=test $F TASK_IDS_FILE=$I/repeat300_mu.json
grun 24 _gx4kr2_bbh100 BENCHMARK=BBH SPLIT=full $F TASK_IDS_FILE=$I/repeat300_bbh.json
wait
G=traces/tasks_sweep_gemma-4-26B-A4B-mk
for t in l1r gx4k; do
    cat ${G}_${t}s{0,1}_fr100.jsonl > ${G}_${t}_fr100.jsonl
    cat ${G}_${t}s{0,1,2,3}_mu100.jsonl > ${G}_${t}_mu100.jsonl
    cat ${G}_${t}s{0,1,2,3}_bbh100.jsonl > ${G}_${t}_bbh100.jsonl
done
wc -l ${G}_{l1r,gx4k}_{ma,fr,mu,bbh}100.jsonl ${G}_gx4kr2_{ma,fr,mu,bbh}100.jsonl
sed "s#glob.glob('traces/tasks_sweep_\*.jsonl')#glob.glob('traces/tasks_sweep_gemma-4-26B-A4B-mk_*_ma100.jsonl')#" analysis/rescore_math.py > $S/rescore_gemma_mk.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python $S/rescore_gemma_mk.py | tail -1
log "GEMMA MK RUNS DONE (FRAMES/MuSiQue still to be judged)"
