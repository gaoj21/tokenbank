#!/bin/bash
# Compaction under a token budget, on Gemma-4-31B bf16. Queued last.
#
# Mirrors the Qwen compaction grid: accumulate-only against accumulate-then-
# summarize, 4 benchmarks x 3 budgets. The claim it tests is that compaction is
# largely inert under a hard per-task budget, and the sharpest evidence for that
# was the 8k stratum, where the trigger fired on 0 of 3,116 Qwen tasks because
# the budget ran out before the context filled.
#
# The trigger is comparable across models by construction: context_window is
# pinned to 8192 in the arm config and is not read from the server, so
# "prompt exceeds ~4,096 tokens" means the same thing here as it did on Qwen
# even though this endpoint serves a 256k window.
set -e
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] compaction waiting for the earlier chains"
while pgrep -f "_chain_gemma_or" >/dev/null || pgrep -f "_chain_gemma_bcb" >/dev/null; do sleep 60; done
KEY=$(cat /home/j0018/.config/openrouter.key)
ARMS="V5-Accum-Only,V5-Compact-Real"
run() {
  local bench=$1 budget=$2 tag=$3 extra=$4
  if [ -s "tasks_sweep_gemma-4-31B-bf16$tag.jsonl" ]; then
    echo "[$(date '+%F %T')] SKIP  $tag"; return
  fi
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://openrouter.ai/api/v1 VLLM_API_KEY="$KEY" \
    VLLM_MODEL=google/gemma-4-31b-it MODEL_TAG=gemma-4-31B-bf16 \
    LLM_EXTRA_BODY_JSON='{"provider":{"only":["venice"],"allow_fallbacks":false}}' \
    BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=64 SPLIT=test \
    V5=1 CONFIG_ONLY="$ARMS" OUT_TAG=$tag TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
for B in 8000 16000 24000; do run GAIA    $B _gcmp_ga${B%000} GAIA_N=1000; done
for B in 8000 16000 24000; do run MATH    $B _gcmp_ma${B%000} MATH_N=500; done
for B in 8000 16000 24000; do run FRAMES  $B _gcmp_fr${B%000} x=1; done
for B in 8000 16000 24000; do run MUSIQUE $B _gcmp_mu${B%000} MUSIQUE_N=99999; done
echo "[$(date '+%F %T')] GEMMA COMPACTION DONE"
