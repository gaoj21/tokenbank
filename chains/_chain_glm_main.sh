#!/bin/bash
# GLM-5.3-Flash-FP8 main table, third model family, served by sglang at llm3.
#
# Protocol note, and it is not a detail: this model cannot be made to stop thinking.
# enable_thinking=false, thinking.type=disabled, reasoning_effort=none, /nothink in the
# system or user turn and separate_reasoning=false were all tried; every one of them
# generates the identical 80 reasoning tokens on the probe prompt. What those flags change
# is only whether sglang splits the trace out of `content`, and two of them leave a stray
# </think> inside it. We therefore run the default, where `content` is clean and the trace
# lands in `reasoning_content`, which the harness never reads.
#
# The consequence is that this family runs with reasoning ON while Qwen and Gemma ran with
# it off. Reasoning tokens are counted in completion_tokens and so are charged to the task
# budget, which is the regime the paper's limitations section names as untested. Two things
# to check when the results land: whether the governed arm still costs less than the
# baseline (in the pilot it cost more), and whether the confidence gate still separates,
# since logprobs here cover the reasoning tokens rather than the answer.
set -e
cd /home/j0018/Desktop/tokenbank
ARMS="V5-Unconstrained,V5-Unc-Fmt,V5.1-Ab7-NoVerbal"
run() {
  local bench=$1 budget=$2 tag=$3 extra=$4
  [ -s "tasks_sweep_GLM-5.3-Flash$tag.jsonl" ] && { echo "SKIP $tag"; return; }
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm3.ruoyu-wang.com/v1 VLLM_API_KEY=EMPTY \
    VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=GLM-5.3-Flash \
    BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=32 SPLIT=test \
    V5=1 CONFIG_ONLY="$ARMS" OUT_TAG=$tag TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
for B in 24000 16000 8000; do run GAIA    $B _glm_ga${B%000} GAIA_N=1000; done
for B in 24000 16000 8000; do run MATH    $B _glm_ma${B%000} MATH_N=500; done
for B in 24000 16000 8000; do run FRAMES  $B _glm_fr${B%000} x=1; done
for B in 24000 16000 8000; do run MUSIQUE $B _glm_mu${B%000} MUSIQUE_N=99999; done
echo "[$(date '+%F %T')] GLM MAIN TABLE DONE"
