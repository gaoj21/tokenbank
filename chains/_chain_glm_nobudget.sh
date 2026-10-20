#!/bin/bash
# INTERNAL VALIDATION ONLY (GLM is not used in the paper): rerun GLM-5.3-Flash with the settings of the
# earlier 24k records but a 100k safety cap instead of the 24k budget, to check whether team harms
# persist without budget truncation. MuSiQue on a 484-task subset.
cd /home/j0018/Desktop/tokenbank
run() { local t=$1 url=$2; shift 2
  [ -s traces/tasks_sweep_GLM-5.3-Flash$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=GLM-5.3-Flash SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=20 \
    MAX_TOKENS=4096 OUT_TAG=$t BUDGETS=100000 SPLIT=test "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/GLM-5.3-Flash$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t $(wc -l < traces/tasks_sweep_GLM-5.3-Flash$t.jsonl 2>/dev/null)"; }
L3=https://llm3.ruoyu-wang.com/v1; L5=https://llm5.ruoyu-wang.com/v1; I=results/climb_live
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"; P="CONFIG_ONLY=V5-Unc-Fmt"
run _glmnb_climbx_ma100 $L3 BENCHMARK=MATH MATH_N=500 $C &
run _glmnb_gx4k_ma100 $L3 BENCHMARK=MATH MATH_N=500 $P &
run _glmnb_climbx_fr100 $L3 BENCHMARK=FRAMES $C &
run _glmnb_gx4k_fr100 $L5 BENCHMARK=FRAMES $P &
run _glmnb_climbx_mu100 $L5 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $C TASK_IDS_FILE=$I/oss_mu_shard0.json &
run _glmnb_gx4k_mu100 $L5 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $P TASK_IDS_FILE=$I/oss_mu_shard0.json &
wait; echo "[$(date '+%F %T')] GLM NOBUDGET DONE"
