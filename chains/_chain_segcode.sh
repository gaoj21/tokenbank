#!/bin/bash
# Segment accounting on the two coding benchmarks.
#
# The four existing benchmarks put raw tool output at 0.4% of processed
# characters, so the mechanism that bounds it has never met a workload where it
# matters, and the plan's 40.1% share has only been shown on question-answering.
# Coding is where both should change: the executor runs code, reads tracebacks
# and retries, so tool output should grow, and it is worth checking whether the
# plan still dominates when the task is to write a function rather than answer.
set -e
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 cfg=$2 tag=$3 extra=$4
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
    BENCHMARK=$bench BUDGETS=16000 SEEDS=42 MAX_WORKERS=16 SPLIT=full \
    V5=1 V5_MAX_ROUNDS=20 CONFIG_ONLY="$cfg" OUT_TAG=$tag TEMPERATURE=0 \
    SEGMENT_ACCOUNTING=1 DUMP_RAW_ANSWER=1 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
run BCB  BCB-Naive  _seg_BCB  BCB_N=500
run SWEO SWEO-Naive _seg_SWEO SWE_N=500
echo "[$(date '+%F %T')] SEGCODE DONE"
