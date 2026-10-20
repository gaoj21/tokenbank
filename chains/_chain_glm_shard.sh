#!/bin/bash
# One shard of the GLM table. Usage: _chain_glm_shard.sh <endpoint-url> <model-id> <tag> <cells...>
#
# The three endpoints are the same deployment: get_server_info reports tp_size=8,
# kv_cache_dtype=bfloat16 and dtype=auto on all three, and with reasoning_effort=low they
# return token-identical generations on a probe prompt (101 reasoning / 129 completion, the
# reasoning text matching word for word). The -FP8 suffix in llm3's served_model_name is a
# naming choice; the weights are FP8 in all three because that is what the published
# checkpoint is. Cells are therefore comparable across shards.
#
# Each cell is written by exactly one shard, so no file is appended to by two processes.
set -eu
cd /home/j0018/Desktop/tokenbank
URL=$1; MODEL=$2; TAG=$3; shift 3
THREE="V5-Unconstrained,V5-Unc-Fmt,V5.2-ReasonLadder"
declare -A BENCH=( [ga]=GAIA [ma]=MATH [fr]=FRAMES [mu]=MUSIQUE )
declare -A EXTRA=( [ga]=GAIA_N=1000 [ma]=MATH_N=500 [fr]=x=1 [mu]=MUSIQUE_N=99999 )
for cell in "$@"; do
  ds=${cell%%[0-9]*}; b=${cell#$ds}
  if .venv/bin/python _glm_status.py 2>/dev/null | grep -qE "^${cell}[[:space:]]+DONE"; then
    echo "[$(date '+%F %T')] SKIP $cell"; continue
  fi
  echo "[$(date '+%F %T')] START $cell on $TAG"
  env OPENAI_BASE_URL="$URL" VLLM_API_KEY=EMPTY \
    VLLM_MODEL="$MODEL" MODEL_TAG=GLM-5.3-Flash \
    BENCHMARK=${BENCH[$ds]} BUDGETS=${b}000 SEEDS=42 MAX_WORKERS=32 SPLIT=test \
    V5=1 CONFIG_ONLY="$THREE" OUT_TAG=_glm4_$cell TEMPERATURE=0 ${EXTRA[$ds]} \
    .venv/bin/python -u _budget_sweep_v2.py > logs/_glm4_$cell.log 2>&1
  echo "[$(date '+%F %T')] DONE  $cell on $TAG"
done
echo "[$(date '+%F %T')] SHARD $TAG DONE"
