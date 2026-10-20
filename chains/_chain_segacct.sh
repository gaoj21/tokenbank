#!/bin/bash
# Direct measurement of what the pipeline's tokens are actually carrying.
#
# The paper's motivation currently says the executor accounts for 50-56% of
# tokens and "the rest is coordination state". That number was measured on
# Qwen3.6 by attributing tokens to the AGENT that was called, which cannot
# answer the question: the plan and the reviewer's critique are re-sent inside
# the executor's own prompt every round and are billed to the executor. On
# Qwen3.8 that same role attribution puts the executor at 73.8% on FRAMES, so
# the claim does not even survive the model change as stated.
#
# SEGMENT_ACCOUNTING=1 decomposes every prompt by what each character carries
# (question / plan / critique / prior answer / tool output / scaffold), parsing
# the exact format strings the pipeline constructs. Run on the naive baseline,
# which is the pipeline the motivation is about, across all four benchmarks.
set -e
cd /home/j0018/Desktop/tokenbank
for spec in "GAIA 122 GAIA_N=1000" "MATH 400 MATH_N=500" "FRAMES 660 x=1" "MUSIQUE 1934 MUSIQUE_N=99999"; do
  set -- $spec; b=$1; extra=$3
  echo "[$(date '+%F %T')] START seg_$b"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
    BENCHMARK=$b BUDGETS=16000 SEEDS=42 MAX_WORKERS=20 SPLIT=test \
    V5=1 CONFIG_ONLY="V5-Unc-Fmt" OUT_TAG=_seg_$b TEMPERATURE=0 \
    SEGMENT_ACCOUNTING=1 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/_seg_$b.log 2>&1
  echo "[$(date '+%F %T')] DONE  seg_$b"
done
echo "[$(date '+%F %T')] SEGACCT ALL DONE"
