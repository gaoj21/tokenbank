#!/bin/bash
# Isolates the verbalized ESCALATE channel on Qwen3.8 (paper section 5.4).
# Ab7 == V5.1-NoPool with m2_verbal off, so the ladder is gated purely by the
# log-probability threshold. Two datasets where the gate's signal differs most:
# MATH (AUC 0.745 at the L0 probe) and MuSiQue (AUC 0.524, i.e. none).
set -e
cd /home/j0018/Desktop/tokenbank
for spec in "MATH 400 _q38_ab7_ma MATH_N=500" "MUSIQUE 1934 _q38_ab7_mu MUSIQUE_N=99999"; do
  set -- $spec; bench=$1; tag=$3; extra=$4
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
    BENCHMARK=$bench BUDGETS=24000 SEEDS=42 MAX_WORKERS=20 SPLIT=test \
    V5=1 CONFIG_ONLY="V5.1-Ab7-NoVerbal" OUT_TAG=$tag TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
done
echo "[$(date '+%F %T')] VERBAL ALL DONE"
