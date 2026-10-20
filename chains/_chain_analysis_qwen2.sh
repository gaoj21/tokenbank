#!/bin/bash
# Follow-ups queued after the first Qwen chain.
#  deploy control : Naive+Fmt at the default 1024 cap on this H20 deployment, MATH. If it lands
#                   near the main table's 57.2%, the 75.0% at 4096 is the cap, not the move.
#  allesc 4096    : the all-escalate run repeated at MAX_TOKENS=4096, so Qwen's L2 is not
#                   truncated and is comparable with GLM's and DeepSeek's.
cd /home/j0018/Desktop/tokenbank
while ! grep -q "QWEN ANALYSIS ALL DONE" logs/_chain_analysis_qwen.log 2>/dev/null; do sleep 60; done
TAG=Qwen3.8-27B
base() { env OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=qwen3.8-27b-fp8 \
  MODEL_TAG=$TAG SEEDS=42 MAX_WORKERS=512 TEMPERATURE=0 V5=1 "$@"; }
run() { local t=$1; shift
  [ -s "traces/tasks_sweep_$TAG$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"; base OUT_TAG=$t "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
run _qx1k_ma24 BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt
for B in MATH:ma:MATH_N=500 GAIA:ga:GAIA_N=1000 FRAMES:fr:x=1 MUSIQUE:mu:MUSIQUE_N=99999; do
  IFS=: read bench t extra <<< "$B"
  run _qx4ka_${t}24 BENCHMARK=$bench BUDGETS=24000 SPLIT=test $extra MAX_TOKENS=4096 CONFIG_ONLY=V5.1-Ab7-AllEsc
done
echo "[$(date '+%F %T')] QWEN FOLLOWUP ALL DONE"
