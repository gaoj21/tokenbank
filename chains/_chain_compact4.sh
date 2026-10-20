#!/bin/bash
# Compaction baseline on the four main-table benchmarks.
#
# This arm has never been run here with a correct trigger. The first version
# compared an agent's CUMULATIVE SPEND against its budget quota, so it fired on
# exactly the runs already headed for exhaustion and its numbers meant nothing;
# after the fix (prompt occupancy of a context window) only SWE-bench was
# re-run. The claim that "compaction has no operating regime on these tasks" was
# therefore inferred from non-accumulating runs, never measured. It may well be
# wrong: segment accounting puts GAIA at ~34k characters per task, so an
# accumulating trajectory can reach the 4,096-token trigger.
#
# V5-Compact-Real accumulates the trajectory and calls an LLM summarizer, paid
# from the same budget, once the prompt exceeds half of an 8k context window.
# 16k first, since that is the budget where the cap binds without starving the
# summarizer; the other two follow.
set -e
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] compact4 waiting for the running sweep"
while pgrep -f "_chain_bcb" >/dev/null; do sleep 60; done
run() {
  local bench=$1 budget=$2 tag=$3 extra=$4
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
    BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=32 SPLIT=test \
    V5=1 CONFIG_ONLY="V5-Compact-Real" OUT_TAG=$tag TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
for b in 16000 8000 24000; do
  k=$((b/1000))k
  run GAIA    $b _cmp_ga$k GAIA_N=1000
  run MATH    $b _cmp_ma$k MATH_N=500
  run FRAMES  $b _cmp_fr$k x=1
  run MUSIQUE $b _cmp_mu$k MUSIQUE_N=99999
done
echo "[$(date '+%F %T')] COMPACT4 ALL DONE"
