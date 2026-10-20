#!/bin/bash
# Qwen3.8-flash with reasoning ON, to separate "reasoning is on" from "it is GLM".
#
# Every other family in this project runs with reasoning off, and GLM runs with it on
# because it cannot be turned off, so those two facts are perfectly confounded: the claim
# that prefix-only governance loses its token advantage under reasoning rests on a single
# model. This family can toggle it, and the toggle verifiably works (enable_thinking=false
# zeroes the reasoning field and drops a probe from 44 to 7 completion tokens), so running
# it on gives a within-family comparison against the Qwen table already on disk.
#
# The reasoning rung is the same configuration as on GLM: reasoning_effort=low on the levels
# the gate keeps. Measured on 40 MATH items here it cuts reasoning 1,276 -> 588 tokens and
# total generation 1,408 -> 904 at identical accuracy (32/40 either way), the same shape it
# has on GLM. thinking_budget compresses harder but costs accuracy (29/40 at 128), so it is
# not used.
#
# Concurrency 96. A ramp found no errors up to 96 and one connection error at 128. Three
# sustained rounds at 64 fell from 2,865 to 632 tok/s, which looks like a bucket that drains
# rather than a per-request limit; if that is what it is, raising the concurrency buys
# nothing and the observed rate here will say so within the first cell.
#
# One thing to declare when writing this up: logprobs here cover `content` only, not the
# reasoning trace (192 covered tokens against 409 reasoning ones on a probe), where on GLM
# they start at the first reasoning token. phi therefore measures the answer on this family
# and the trace on that one.
set -e
cd /home/j0018/Desktop/tokenbank
ARMS="V5-Unc-Fmt,V5.1-Ab7-NoVerbal,V5.2-ReasonLadder"
BCBARMS="BCB-Naive,BCB-Ab7,BCB-ReasonLadder"
run() {
  local bench=$1 budget=$2 tag=$3 extra=$4
  [ -s "tasks_sweep_Qwen3.8-flash-think$tag.jsonl" ] && { echo "SKIP $tag"; return; }
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
    VLLM_API_KEY="$(cat /home/j0018/.config/dashscope.key)" \
    VLLM_MODEL=qwen3.8-flash MODEL_TAG=Qwen3.8-flash-think \
    LLM_EXTRA_BODY_JSON='{"enable_thinking":true}' \
    BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=96 SPLIT=test \
    V5=1 CONFIG_ONLY="$ARMS" OUT_TAG=$tag TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
run_bcb() {
  local budget=$1 tag=$2
  [ -s "tasks_sweep_Qwen3.8-flash-think$tag.jsonl" ] && { echo "SKIP $tag"; return; }
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
    VLLM_API_KEY="$(cat /home/j0018/.config/dashscope.key)" \
    VLLM_MODEL=qwen3.8-flash MODEL_TAG=Qwen3.8-flash-think \
    LLM_EXTRA_BODY_JSON='{"enable_thinking":true}' \
    BENCHMARK=BCB BCB_N=500 BUDGETS=$budget SEEDS=42 MAX_WORKERS=96 SPLIT=full \
    V5=1 V5_MAX_ROUNDS=5 CONFIG_ONLY="$BCBARMS" OUT_TAG=$tag TEMPERATURE=0 \
    DUMP_RAW_ANSWER=1 \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
# BigCodeBench first: on GLM this is where the reasoning rung showed its strongest result,
# ahead of prefix-only governance at every budget on accuracy as well as tokens, so it is
# the cell most worth having on a second thinking model. Its arms follow the BCB convention
# and its budgets are 4k/8k/16k, since 24k does not bind on this benchmark.
for B in 4000 8000 16000; do
  run_bcb $B _qtbcb${B%000}k
done
for B in 24000 16000 8000; do run MATH $B _qt_ma${B%000} MATH_N=500; done
for B in 24000 16000 8000; do run GAIA $B _qt_ga${B%000} GAIA_N=1000; done
echo "[$(date '+%F %T')] QWEN THINKING DONE"
