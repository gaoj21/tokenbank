#!/bin/bash
# GLM budget accounting for Section 4: Naive+Fmt at 16k, cap 4096, SEGMENT_ACCOUNTING=1, full
# benchmark sets, plus BCB-Naive at 16k. Same configuration as the DeepSeek 16k reruns.
cd /home/j0018/Desktop/tokenbank
T=GLM-5.3-Flash
run() { local t=$1 url=$2 model=$3; shift 3
  [ -s "traces/tasks_sweep_$T$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 \
      MAX_WORKERS=32 MAX_TOKENS=4096 SEGMENT_ACCOUNTING=1 OUT_TAG=$t "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
run _glmseg_ma16 https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 BENCHMARK=MATH MATH_N=500 BUDGETS=16000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt &
run _glmseg_ga16 https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 BENCHMARK=GAIA GAIA_N=1000 BUDGETS=16000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt &
run _glmseg_fr16 https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash BENCHMARK=FRAMES BUDGETS=16000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt &
run _glmseg_mu16 https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=16000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt &
run _glmseg_bcb16k https://llm57.jierungogogo.com/v1 zai-org/GLM-5.3-Flash BENCHMARK=BCB BCB_N=500 BUDGETS=16000 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1 CONFIG_ONLY=BCB-Naive &
wait; echo "[$(date '+%F %T')] GLM SEG DONE"
