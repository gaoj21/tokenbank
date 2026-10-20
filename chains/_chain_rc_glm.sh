#!/bin/bash
# Feasibility test for the runtime per-call cap: GLM, MATH and GAIA, at 8k and 24k, where the
# fixed-cap crossover is largest. Baselines already exist: Naive+Fmt at cap 1024 (glm4_*) and
# cap 4096 (glmx4k_*24, glmr4k_*8).
cd /home/j0018/Desktop/tokenbank
T=GLM-5.3-Flash
run() { local t=$1 url=$2 model=$3; shift 3
  [ -s traces/tasks_sweep_$T$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=32 \
    OUT_TAG=$t CONFIG_ONLY=RC-Adaptive,RC-CapOnly "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
run _glmrc_ma24 https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test &
run _glmrc_ma8  https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 BENCHMARK=MATH MATH_N=500 BUDGETS=8000 SPLIT=test &
run _glmrc_ga24 https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SPLIT=test &
run _glmrc_ga8  https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash BENCHMARK=GAIA GAIA_N=1000 BUDGETS=8000 SPLIT=test &
wait; echo "[$(date '+%F %T')] RC GLM DONE"
