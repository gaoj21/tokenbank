#!/bin/bash
# Qwen3.8-flash, full re-run, mirroring the GLM configuration exactly.
#
# Why re-run rather than extend: the earlier pass for this family went through DashScope,
# where logprobs covered `content` only and the reasoning trace was excluded from phi, and
# it ended with the account suspended mid-MuSiQue, leaving three cells of rows that carried
# no tokens at all. This deployment is a vLLM behind an SSH tunnel, where the trace is not
# split out of `content` and the logprobs therefore cover reasoning and answer together, so
# phi is a different quantity. Mixing the two passes would put two definitions of the gate's
# own input in one row. The DashScope files are under _aborted/qt_dashscope/.
#
# Reasoning is on and cannot be turned off here: enable_thinking true, false and absent all
# return the same 34 reasoning tokens on a probe. enable_thinking is still sent, to record
# the intent in the trace rather than to change anything.
#
# Arms and budgets follow GLM: three arms on the main table, the BCB convention on
# BigCodeBench with its own budgets, and the Ab7 ablation on MATH only.
#
# No reasoning rung on this family. The graded settings are named but not acted on here:
# low, medium and xhigh all produce 524 to 605 reasoning tokens over the same 40 problems,
# and a first pass at BCB 4k put ReasonLadder and Ab7 within 1.4% of each other. The only
# control the server honours is thinking_token_budget, which is a hard cut mid-thought
# rather than an instruction to think less, and it behaves like one: capped at this
# benchmark's P25 of 312, reasoning falls from 682 to 272 but 28 of 40 samples hit the cap
# and the number that produce an answer at all falls from 36 to 29, taking accuracy from
# 28/40 to 22/40. On GLM the graded setting cuts the per-call median from 61 to 15 with
# accuracy unchanged, so the two are not the same mechanism and putting them in one column
# would misreport what was measured. This family therefore runs the three prefix-only arms
# and answers one question: whether bounding the carried prefix still pays when reasoning
# is on. The reasoning-rung evidence stays with GLM alone.
#
# Concurrency 64. A ramp from 256 down found zero errors at 256 and at 64, and connection
# errors through the 96-192 band, which is the tunnel rather than the server; 64 gives
# 3,285 tok/s against a 5,024 peak at 128 and does not drop calls.
set -e
cd /home/j0018/Desktop/tokenbank
TAG=Qwen3.8-flash-vllm
URL=http://127.0.0.1:18000/v1
THREE="V5-Unconstrained,V5-Unc-Fmt,V5.1-Ab7-NoVerbal"
BCBARMS="BCB-Naive,BCB-Ab7"
base() {
  env OPENAI_BASE_URL=$URL VLLM_API_KEY=EMPTY VLLM_MODEL=qwen3.8-flash MODEL_TAG=$TAG \
      LLM_EXTRA_BODY_JSON='{"enable_thinking":true}' \
      SEEDS=42 MAX_WORKERS=64 TEMPERATURE=0 V5=1 "$@"
}
run() {
  local bench=$1 budget=$2 tag=$3 extra=$4
  [ -s "tasks_sweep_$TAG$tag.jsonl" ] && { echo "SKIP $tag"; return; }
  echo "[$(date '+%F %T')] START $tag"
  base BENCHMARK=$bench BUDGETS=$budget SPLIT=test CONFIG_ONLY="$THREE" \
       OUT_TAG=$tag $extra .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
run_bcb() {
  local budget=$1 tag=$2
  [ -s "tasks_sweep_$TAG$tag.jsonl" ] && { echo "SKIP $tag"; return; }
  echo "[$(date '+%F %T')] START $tag"
  base BENCHMARK=BCB BCB_N=500 BUDGETS=$budget SPLIT=full V5_MAX_ROUNDS=5 \
       CONFIG_ONLY="$BCBARMS" OUT_TAG=$tag DUMP_RAW_ANSWER=1 \
       .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
for B in 4000 8000 16000; do run_bcb $B _qv_bcb${B%000}k; done
for B in 24000 16000 8000; do run MATH    $B _qv_ma${B%000} MATH_N=500; done
for B in 24000 16000 8000; do run GAIA    $B _qv_ga${B%000} GAIA_N=1000; done
for B in 24000 16000 8000; do run FRAMES  $B _qv_fr${B%000} x=1; done
for B in 24000 16000 8000; do run MUSIQUE $B _qv_mu${B%000} MUSIQUE_N=99999; done
echo "[$(date '+%F %T')] QWEN VLLM MAIN DONE"
