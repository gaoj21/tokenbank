#!/bin/bash
# Fill in the mechanism decomposition at 8k and 16k. The table currently has one budget,
# where no increment is individually significant at n=400; three budget points give the
# chain of round counts, reach and discard a trend to be read against rather than a single
# noisy column. Routing is off in these arms so every item runs the full loop, which is what
# makes the increments comparable.
set -e
cd /home/j0018/Desktop/tokenbank
for b in 8000 16000; do
  tag=_qdec_ma${b%000}
  [ -s "traces/tasks_sweep_Qwen3.8-27B$tag.jsonl" ] && { echo "SKIP $tag"; continue; }
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm5.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=qwen3.8-27b-fp8 MODEL_TAG=Qwen3.8-27B \
    BENCHMARK=MATH MATH_N=500 BUDGETS=$b SEEDS=42 MAX_WORKERS=32 SPLIT=test \
    V5=1 CONFIG_ONLY="V5.1-Ab6-NoM2NoM1,V5.1-Ab3-NoM2" OUT_TAG=$tag TEMPERATURE=0 \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
done
echo "[$(date '+%F %T')] QWEN DECOMP DONE"
