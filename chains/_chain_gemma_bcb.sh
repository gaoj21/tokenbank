#!/bin/bash
# BigCodeBench on Gemma-4-31B bf16, queued behind the main-table chain.
#
# With SWE-bench dropped, BigCodeBench carries the paper's entire code claim,
# so leaving it on one model family only is a gap. It is also cheap here:
# roughly 5M tokens and $0.79 at Venice rates, against $25 for the main table.
#
# Arms follow the Qwen BCB run, not the main table: the answer is a code block
# judged by executing the task's own unittest suite, so the ANSWER: format
# alignment that Naive+Fmt exists to control for does not apply.
# DUMP_RAW_ANSWER=1 is required; the sandboxed judge runs offline afterwards.
set -e
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] BCB waiting for the main-table chain"
while pgrep -f "_chain_gemma_or" >/dev/null; do sleep 60; done
KEY=$(cat /home/j0018/.config/openrouter.key)
for b in 4000 8000 16000; do
  tag="_gbcb${b%000}k"
  if [ -s "tasks_sweep_gemma-4-31B-bf16$tag.jsonl" ]; then
    echo "[$(date '+%F %T')] SKIP  $tag"; continue
  fi
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://openrouter.ai/api/v1 VLLM_API_KEY="$KEY" \
    VLLM_MODEL=google/gemma-4-31b-it MODEL_TAG=gemma-4-31B-bf16 \
    LLM_EXTRA_BODY_JSON='{"provider":{"only":["venice"],"allow_fallbacks":false}}' \
    BENCHMARK=BCB BCB_N=500 BUDGETS=$b SEEDS=42 MAX_WORKERS=64 SPLIT=full \
    V5=1 V5_MAX_ROUNDS=5 CONFIG_ONLY="BCB-Naive,BCB-Ours,BCB-Ab7" \
    OUT_TAG=$tag TEMPERATURE=0 DUMP_RAW_ANSWER=1 \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
done
echo "[$(date '+%F %T')] GEMMA BCB DONE (judge separately with _bcb_judge.py)"
