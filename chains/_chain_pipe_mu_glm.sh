#!/bin/bash
# GLM contemporaneous pipeline on the MuSiQue live test half, in three shards; merged at the end.
cd /home/j0018/Desktop/tokenbank
run() { local T=$1 t=$2 url=$3 model=$4 w=$5 ids=$6; shift 6
  echo "[$(date '+%F %T')] START $T$t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    MAX_TOKENS=4096 TASK_IDS_FILE=$ids OUT_TAG=$t CONFIG_ONLY=V5-Unc-Fmt "$@" \
    .venv/bin/python -u _budget_sweep_v2.py > logs/${T}$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
G=GLM-5.3-Flash; I=results/climb_live; M="BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test"
run $G _pipe3_mu24s0 https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash 32 $I/pipe_mu_shard0.json $M &
run $G _pipe3_mu24s1 https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 $I/pipe_mu_shard1.json $M &
run $G _pipe3_mu24s2 https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 32 $I/pipe_mu_shard2.json $M &
wait
cat traces/tasks_sweep_${G}_pipe3_mu24s{0,1,2}.jsonl > traces/tasks_sweep_${G}_pipe3_mu24.jsonl
echo "[$(date '+%F %T')] PIPE MU GLM DONE"
