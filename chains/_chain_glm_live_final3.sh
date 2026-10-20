#!/bin/bash
# Third MuSiQue live shard (322 tasks) on llm3 after the FRAMES live run, then merge all three.
cd /home/j0018/Desktop/tokenbank
G=GLM-5.3-Flash; TM=-0.415
until [ -s traces/tasks_sweep_${G}_climb3_fr24.jsonl ]; do sleep 20; done
echo "[$(date '+%F %T')] FRAMES live done APIERR=$(grep -c APIERR logs/${G}_climb3_fr24.log)"
echo "[$(date '+%F %T')] START _climb3_mu24x2 (llm3) theta=$TM"
env OPENAI_BASE_URL=https://llm3.ruoyu-wang.com/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=$G SEEDS=42 TEMPERATURE=0 V5=1 \
  MAX_WORKERS=32 MAX_TOKENS=4096 OUT_TAG=_climb3_mu24x2 CONFIG_ONLY=CLIMB-Live THETA_LIVE=$TM TASK_IDS_FILE=results/climb_live/live_mu_x2.json \
  BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test .venv/bin/python -u _budget_sweep_v2.py > logs/${G}_climb3_mu24x2.log 2>&1
until [ -s traces/tasks_sweep_${G}_climb3_mu24x0.jsonl ] && [ -s traces/tasks_sweep_${G}_climb3_mu24x1.jsonl ]; do sleep 20; done
cat traces/tasks_sweep_${G}_climb3_mu24x{0,1,2}.jsonl > traces/tasks_sweep_${G}_climb3_mu24.jsonl
echo "[$(date '+%F %T')] MuSiQue live tasks=$(wc -l < traces/tasks_sweep_${G}_climb3_mu24.jsonl) APIERR=$(cat logs/${G}_climb3_mu24x{0,1,2}.log | grep -c APIERR)"
echo "[$(date '+%F %T')] GLM LIVE FINAL DONE"
