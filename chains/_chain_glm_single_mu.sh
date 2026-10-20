#!/bin/bash
# GLM single-call MuSiQue on FP8 (appendix reference), three shards, after the live runs finish.
cd /home/j0018/Desktop/tokenbank
until grep -q "GLM FP8 ALL DONE" logs/_chain_glm_fp8_queue2.log; do sleep 30; done
k=0; for h in llm3 llm4 llm5; do
  env OPENAI_BASE_URL=https://$h.ruoyu-wang.com/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=GLM-5.3-Flash SEEDS=42 TEMPERATURE=0 V5=1 \
    MAX_WORKERS=32 MAX_TOKENS=4096 OUT_TAG=_glmfp8s_mu24x$k CONFIG_ONLY=V5-Single-Fmt-4k TASK_IDS_FILE=results/climb_live/mu_single_$k.json \
    BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test .venv/bin/python -u _budget_sweep_v2.py > logs/GLM-5.3-Flash_glmfp8s_mu24x$k.log 2>&1 &
  k=$((k+1)); done
wait
cat traces/tasks_sweep_GLM-5.3-Flash_glmfp8s_mu24x{0,1,2}.jsonl > traces/tasks_sweep_GLM-5.3-Flash_glmfp8s_mu24.jsonl
echo "[$(date '+%F %T')] SINGLE MU DONE tasks=$(wc -l < traces/tasks_sweep_GLM-5.3-Flash_glmfp8s_mu24.jsonl) APIERR=$(cat logs/GLM-5.3-Flash_glmfp8s_mu24x*.log | grep -c APIERR)"
