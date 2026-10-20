#!/bin/bash
# GLM-5.3-Flash-FP8 full table, three arms.
#
# This model cannot be made to stop thinking. enable_thinking=false,
# thinking.type=disabled, reasoning_effort=none, /nothink in either turn and
# separate_reasoning=false all produce the identical trace; they change only whether sglang
# splits it out of `content`, and two of them leave a stray </think> inside it. So the
# family runs with reasoning on, and the term that grows is the per-call generation rather
# than the carried prefix.
#
# The governed arm is therefore V5.2-ReasonLadder, which gives the levels the gate keeps a
# cheaper reasoning setting. V5.1-Ab7-NoVerbal, the configuration used for the two families
# whose thinking is off, is not run here as a main-table arm: it leaves the dominant cost
# term ungoverned, and what that costs is already measured on GAIA at 24k and 16k, where it
# spent -2% and +2% against the baseline while the same arm saves 24-67% elsewhere. Those
# two cells are the ablation; re-running the most expensive arm on every benchmark to
# re-confirm a negative is not worth the compute.
set -e
cd /home/j0018/Desktop/tokenbank
THREE="V5-Unconstrained,V5-Unc-Fmt,V5.2-ReasonLadder"
run() {
  local bench=$1 budget=$2 tag=$3 arms=$4 extra=$5
  if .venv/bin/python _glm_status.py 2>/dev/null | grep -qE "^${tag#_glm4_}[[:space:]]+DONE"; then
    echo "[$(date '+%F %T')] SKIP  $tag"; return
  fi
  echo "[$(date '+%F %T')] START $tag [$arms]"
  env OPENAI_BASE_URL=https://llm3.ruoyu-wang.com/v1 VLLM_API_KEY=EMPTY \
    VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=GLM-5.3-Flash \
    BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=32 SPLIT=test \
    V5=1 CONFIG_ONLY="$arms" OUT_TAG=$tag TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
# ga24 already holds the two baseline arms from the interrupted run; add only the third.
run GAIA 24000 _glm4_ga24 "V5.2-ReasonLadder" GAIA_N=1000   # two baseline arms already on disk
for B in 16000 8000; do run GAIA $B _glm4_ga${B%000} "$THREE" GAIA_N=1000; done
for B in 24000 16000 8000; do run MATH    $B _glm4_ma${B%000} "$THREE" MATH_N=500; done
for B in 24000 16000 8000; do run FRAMES  $B _glm4_fr${B%000} "$THREE" x=1; done
for B in 24000 16000 8000; do run MUSIQUE $B _glm4_mu${B%000} "$THREE" MUSIQUE_N=99999; done
echo "[$(date '+%F %T')] GLM ALL DONE"
