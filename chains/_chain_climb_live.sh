#!/bin/bash
# CLIMB live run on the test halves. Thresholds certified offline on the calibration halves
# (results/climb_live_plan.json, eps = 5%, delta = 0.1). DeepSeek GAIA certified 'direct',
# so it runs the pipeline and needs no live run. Cheap-rung settings match the offline records
# the thresholds were certified on: GLM reasoning_effort low; DeepSeek thinking off.
cd /home/j0018/Desktop/tokenbank
run() { local T=$1 t=$2 url=$3 model=$4 w=$5 th=$6 ids=$7; shift 7
  [ -s traces/tasks_sweep_$T$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $T$t theta=$th"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    MAX_TOKENS=4096 THETA_LIVE=$th TASK_IDS_FILE=$ids OUT_TAG=$t CONFIG_ONLY=CLIMB-Live "$@" \
    .venv/bin/python -u _budget_sweep_v2.py > logs/${T}$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
G=GLM-5.3-Flash; I=results/climb_live
run $G _climb_ma24 https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 -0.02 $I/test_ids_GLM_MATH.json BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test &
run $G _climb_ga24 https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 -0.055 $I/test_ids_GLM_GAIA.json BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SPLIT=test &
run $G _climb_fr24 https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 -0.085 $I/test_ids_GLM_FRAMES.json BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test &
run $G _climb_mu24 https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 -0.19 $I/test_ids_GLM_MuSiQue.json BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test &
D=DeepSeek-V4.1-Flash; U=http://127.0.0.1:18000/v1
run $D _climb_ma24 $U deepseek-v4.1-flash 40 -0.05 $I/test_ids_DeepSeek_MATH.json BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test L01_THINKING_OFF=1 &
run $D _climb_fr24 $U deepseek-v4.1-flash 40 -0.03 $I/test_ids_DeepSeek_FRAMES.json BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test L01_THINKING_OFF=1 &
run $D _climb_mu24 $U deepseek-v4.1-flash 40 -0.08 $I/test_ids_DeepSeek_MuSiQue.json BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test L01_THINKING_OFF=1 &
wait; echo "[$(date '+%F %T')] CLIMB LIVE DONE"
