#!/bin/bash
# Gemma MATH/BBH reference reruns through OpenRouter (Makora) with the fixed reviewer verdict; same settings as
# _chain_gemma_mk.sh. New tags gx5k / gx5kr2. Then rescore MATH from snapshots.
cd /home/j0018/Desktop/tokenbank
log() { echo "[$(date '+%F %T')] $*"; }
I=results/climb_live; R=chains/_gemma_mk_run.sh; export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
B="BUDGETS=10000000 L01_MAX_TOKENS=8192 TOP_LOGPROBS=0 MAX_TOKENS=16384 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0 CONFIG_ONLY=V5-Unc-Fmt"
g() { MAX_WORKERS=$1 bash $R "${@:2}" $B & }
g 40 _gx5k_ma100 BENCHMARK=MATH MATH_N=500 SPLIT=test
g 32 _gx5kr2_ma100 BENCHMARK=MATH MATH_N=500 SPLIT=test TASK_IDS_FILE=$I/repeat300_ma.json
g 56 _gx5k_bbh100 BENCHMARK=BBH SPLIT=full
g 32 _gx5kr2_bbh100 BENCHMARK=BBH SPLIT=full TASK_IDS_FILE=$I/repeat300_bbh.json
wait
G=traces/tasks_sweep_gemma-4-26B-A4B-mk; wc -l ${G}_gx5k*_{ma,bbh}100.jsonl
S=/tmp/claude-1001/-home-j0018-Desktop-tokenbank/f14fef7e-adf9-443b-9307-5c07475168b4/scratchpad
sed "s#glob.glob('traces/tasks_sweep_\*.jsonl')#glob.glob('traces/tasks_sweep_gemma-4-26B-A4B-mk_gx5k*_ma100.jsonl')#" analysis/rescore_math.py > $S/rescore_gx5k_mk.py
.venv/bin/python $S/rescore_gx5k_mk.py | tail -2
log "GEMMA MK RERUN DONE"
