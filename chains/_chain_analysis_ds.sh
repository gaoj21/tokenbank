#!/bin/bash
# Analysis-paper experiments on DeepSeek-V4.1-Flash, same deployment and dataset settings as
# the main table (_chain_deepseek.sh).
#  single : one Executor call, format instruction, default reasoning, at 1k and 2k caps. The
#           budget never binds on one call, so one budget suffices.
#  allesc : the governed arm with gates that never pass, so every item records all three
#           rungs. 24k only; this is the counterfactual for the frontier and the oracle router.
cd /home/j0018/Desktop/tokenbank
TAG=DeepSeek-V4.1-Flash
base() { env OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=deepseek-v4.1-flash \
  MODEL_TAG=$TAG SEEDS=42 MAX_WORKERS=256 TEMPERATURE=0 V5=1 L01_THINKING_OFF=1 "$@"; }
run() { local t=$1; shift
  [ -s "traces/tasks_sweep_$TAG$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"; base OUT_TAG=$t "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
for B in MATH:ma:MATH_N=500 GAIA:ga:GAIA_N=1000 FRAMES:fr:x=1 MUSIQUE:mu:MUSIQUE_N=99999; do
  IFS=: read bench t extra <<< "$B"
  run _dsx_single_${t}24 BENCHMARK=$bench BUDGETS=24000 SPLIT=test $extra CONFIG_ONLY=V5-Single-Fmt-1k,V5-Single-Fmt-2k
done
run _dsx_bcbsingle16k BENCHMARK=BCB BCB_N=500 BUDGETS=16000 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1 CONFIG_ONLY=BCB-Single-1k,BCB-Single-2k
for B in MATH:ma:MATH_N=500 GAIA:ga:GAIA_N=1000 FRAMES:fr:x=1 MUSIQUE:mu:MUSIQUE_N=99999; do
  IFS=: read bench t extra <<< "$B"
  run _dsx_allesc_${t}24 BENCHMARK=$bench BUDGETS=24000 SPLIT=test $extra CONFIG_ONLY=V5.2-ReasonLadder-AllEsc
done
run _dsx_bcballesc16k BENCHMARK=BCB BCB_N=500 BUDGETS=16000 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1 CONFIG_ONLY=BCB-ReasonLadder-AllEsc
echo "[$(date '+%F %T')] DS ANALYSIS ALL DONE"
