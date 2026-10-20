#!/bin/bash
# Final experiments (list closed 2026-10-03):
#  GLM      : ZEBRA-style static split (SS-Static, SS-Static-Keep) on the four test cells
#  DeepSeek : runtime cap (RC-Adaptive, RC-CapOnly) and static split on the same four cells
# Role shares are per model, from the measured role decomposition.
cd /home/j0018/Desktop/tokenbank
run() { local T=$1 t=$2 url=$3 model=$4 w=$5 arms=$6; shift 6
  [ -s traces/tasks_sweep_$T$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $T$t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    OUT_TAG=$t CONFIG_ONLY=$arms "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
G=GLM-5.3-Flash; GS="SS_P=0.21 SS_E=0.48 SS_R=0.31"
run $G _glmss_ma24 https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 SS-Static,SS-Static-Keep BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test $GS &
run $G _glmss_ma8  https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 SS-Static,SS-Static-Keep BENCHMARK=MATH MATH_N=500 BUDGETS=8000 SPLIT=test $GS &
run $G _glmss_ga24 https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 SS-Static,SS-Static-Keep BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SPLIT=test $GS &
run $G _glmss_ga8  https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 SS-Static,SS-Static-Keep BENCHMARK=GAIA GAIA_N=1000 BUDGETS=8000 SPLIT=test $GS &
D=DeepSeek-V4.1-Flash; DS="SS_P=0.30 SS_E=0.42 SS_R=0.28"; U=http://127.0.0.1:18000/v1
for c in "MATH MATH_N=500 24000:ma24" "MATH MATH_N=500 8000:ma8" "GAIA GAIA_N=1000 24000:ga24" "GAIA GAIA_N=1000 8000:ga8"; do
  set -- ${c%%:*}; t=${c##*:}
  run $D _dsrc_$t $U deepseek-v4.1-flash 96 RC-Adaptive,RC-CapOnly,SS-Static,SS-Static-Keep BENCHMARK=$1 $2 BUDGETS=$3 SPLIT=test $DS &
done
wait; echo "[$(date '+%F %T')] FINAL RC DONE"
