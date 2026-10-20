#!/bin/bash
# Qwen3.8-27B-FP8 analysis experiments, on the remote 8xH20 (data parallel 8), weights
# verified identical to the main table's (config.json and crc32.txt match). Thinking off, as
# in the main table (harness default chat_template_kwargs enable_thinking=false).
#  single : one call, 1k and 2k caps
#  allesc : governed arm (Ab7) with gates that never pass, per-call cap as in the main table
#  4k     : Naive+Fmt with MAX_TOKENS=4096, to check whether the 1024 cap binds on a
#           non-reasoning model the way it did on DeepSeek
cd /home/j0018/Desktop/tokenbank
until curl -s -m 5 http://127.0.0.1:18000/v1/models | grep -q qwen3.8-27b-fp8; do sleep 30; done
TAG=Qwen3.8-27B
base() { env OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=qwen3.8-27b-fp8 \
  MODEL_TAG=$TAG SEEDS=42 MAX_WORKERS=512 TEMPERATURE=0 V5=1 "$@"; }
run() { local t=$1; shift
  [ -s "traces/tasks_sweep_$TAG$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"; base OUT_TAG=$t "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
for B in MATH:ma:MATH_N=500 GAIA:ga:GAIA_N=1000 FRAMES:fr:x=1 MUSIQUE:mu:MUSIQUE_N=99999; do
  IFS=: read bench t extra <<< "$B"
  run _qx_single_${t}24 BENCHMARK=$bench BUDGETS=24000 SPLIT=test $extra CONFIG_ONLY=V5-Single-Fmt-1k,V5-Single-Fmt-2k
done
run _qx_bcbsingle16k BENCHMARK=BCB BCB_N=500 BUDGETS=16000 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1 CONFIG_ONLY=BCB-Single-1k,BCB-Single-2k
for B in MATH:ma:MATH_N=500 GAIA:ga:GAIA_N=1000 FRAMES:fr:x=1 MUSIQUE:mu:MUSIQUE_N=99999; do
  IFS=: read bench t extra <<< "$B"
  run _qx_allesc_${t}24 BENCHMARK=$bench BUDGETS=24000 SPLIT=test $extra CONFIG_ONLY=V5.1-Ab7-AllEsc
  run _qx4k_${t}24 BENCHMARK=$bench BUDGETS=24000 SPLIT=test $extra MAX_TOKENS=4096 CONFIG_ONLY=V5-Unc-Fmt
done
echo "[$(date '+%F %T')] QWEN ANALYSIS ALL DONE"
