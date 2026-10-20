#!/bin/bash
# MuSiQue all-escalate records in three shards per model (same configuration as _chain_climbx_*.sh).
cd /home/j0018/Desktop/tokenbank
run() { local T=$1 t=$2 url=$3 model=$4 w=$5 ids=$6; shift 6
  [ -s traces/tasks_sweep_$T$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $T$t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    MAX_TOKENS=4096 THETA_LIVE=99 OUT_TAG=$t CONFIG_ONLY=CLIMB-Live TASK_IDS_FILE=$ids "$@" \
    .venv/bin/python -u _budget_sweep_v2.py > logs/${T}$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
G=GLM-5.3-Flash; D=DeepSeek-V4.1-Flash; U=http://127.0.0.1:18000/v1; S=results/climb_live
M="BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test"
run $G _climbx_mu24s0 https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 $S/mu_shard0.json $M &
run $G _climbx_mu24s1 https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 $S/mu_shard1.json $M &
run $G _climbx_mu24s2 https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 $S/mu_shard2.json $M &
run $D _climbx_mu24s0 $U deepseek-v4.1-flash 40 $S/mu_shard0.json $M L01_THINKING_OFF=1 &
# the other two DeepSeek shards start once MATH and GAIA have freed server slots
( until grep -q "DONE  DeepSeek-V4.1-Flash_climbx_ma24" logs/_chain_climbx_ds.log && grep -q "DONE  DeepSeek-V4.1-Flash_climbx_ga24" logs/_chain_climbx_ds.log; do sleep 30; done
  run $D _climbx_mu24s1 $U deepseek-v4.1-flash 40 $S/mu_shard1.json $M L01_THINKING_OFF=1 &
  run $D _climbx_mu24s2 $U deepseek-v4.1-flash 40 $S/mu_shard2.json $M L01_THINKING_OFF=1 &
  wait ) &
wait
for T in $G $D; do cat traces/tasks_sweep_${T}_climbx_mu24s{0,1,2}.jsonl > traces/tasks_sweep_${T}_climbx_mu24.jsonl; done
echo "[$(date '+%F %T')] CLIMBX MU DONE"
