#!/bin/bash
# Contemporaneous plain pipeline (V5-Unc-Fmt, no draft) on the live test halves, run right after the
# final live CLIMB runs, so live CLIMB can be compared with a pipeline run under the same serving state.
cd /home/j0018/Desktop/tokenbank
run() { local T=$1 t=$2 url=$3 model=$4 w=$5 ids=$6; shift 6
  [ -s traces/tasks_sweep_$T$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $T$t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    MAX_TOKENS=4096 TASK_IDS_FILE=$ids OUT_TAG=$t CONFIG_ONLY=V5-Unc-Fmt "$@" \
    .venv/bin/python -u _budget_sweep_v2.py > logs/${T}$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
G=GLM-5.3-Flash; D=DeepSeek-V4.1-Flash; U=http://127.0.0.1:18000/v1; I=results/climb_live
run $G _pipe3_ma24 https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 $I/test_ids_GLM_MATH.json BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test &
run $G _pipe3_fr24 https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 $I/test_ids_GLM_FRAMES.json BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test &
run $G _pipe3_mu24 https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 $I/test_ids_GLM_MuSiQue.json BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test &
run $D _pipe3_ma24 $U deepseek-v4.1-flash 40 $I/test_ids_DeepSeek_MATH.json BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test &
run $D _pipe3_fr24 $U deepseek-v4.1-flash 40 $I/test_ids_DeepSeek_FRAMES.json BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test &
run $D _pipe3_mu24 $U deepseek-v4.1-flash 40 $I/test_ids_DeepSeek_MuSiQue.json BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test &
wait; echo "[$(date '+%F %T')] PIPE CONTEMP DONE"
