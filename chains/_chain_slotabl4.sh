#!/bin/bash
# Slot ablation across all four benchmarks.
# Two questions the paper asserts but has not measured generally:
#   1. what share of the carried context each slot actually occupies
#   2. whether removing that slot changes the answer, and in which direction
# The first run covered MATH and GAIA only, and GAIA was underpowered.
set -e
cd /home/j0018/Desktop/tokenbank
for spec in "FRAMES 250" "MUSIQUE 250" "GAIA 122" "MATH 250"; do
  set -- $spec; b=$1; n=$2
  echo "[$(date '+%F %T')] START slotabl_$b"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=Qwen/Qwen3.8-27B BENCHMARK=$b N=$n \
    .venv/bin/python -u _slot_ablation.py > logs/_slotabl4_$b.log 2>&1
  echo "[$(date '+%F %T')] DONE  slotabl_$b"
done
echo "[$(date '+%F %T')] SLOTABL4 ALL DONE"
