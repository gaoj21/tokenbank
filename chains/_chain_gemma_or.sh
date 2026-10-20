#!/bin/bash
# Gemma-4-31B-it cross-model run on OpenRouter, mirroring the Qwen main table
# in full: 4 benchmarks x 3 arms x 3 budgets = 28,044 runs.
#
# Why not the local llama.cpp server: it serves the QAT q4_0 GGUF at ~205 tok/s,
# which puts this table at about 14 days, and a 4-bit quantization is weak
# evidence for a cross-model claim. Venice serves bf16 at ~2,400 tok/s measured
# at 64 concurrent with zero failures.
#
# The provider is pinned and fallbacks are off. OpenRouter otherwise picks a
# different upstream per call, and the providers behind this one model id run it
# at fp4, fp8 or bf16 and disagree on whether logprobs come back at all; the
# confidence gate needs them, and a run whose provider drifts is not a
# controlled comparison. Venice was chosen as the cheapest bf16 endpoint that
# returned logprobs on every call under load.
#
# MODEL_TAG is gemma-4-31B-bf16, deliberately distinct from the older
# gemma-4-31B-it traces, which came from the q4_0 build and must not be pooled
# with these.
#
# Cheapest benchmark first, so the table fills progressively and the expensive
# tail (MuSiQue, roughly $23 of the ~$37 total) comes last.
set -e
cd /home/j0018/Desktop/tokenbank
KEY=$(cat /home/j0018/.config/openrouter.key)
ARMS="V5-Unconstrained,V5-Unc-Fmt,V5.1-Ab7-NoVerbal"
run() {
  local bench=$1 budget=$2 tag=$3 extra=$4
  # Resume: a cell whose trace already exists is finished; re-running it would
  # append a second copy of every task to the same file.
  if [ -s "tasks_sweep_gemma-4-31B-bf16$tag.jsonl" ]; then
    echo "[$(date '+%F %T')] SKIP  $tag (already done)"; return
  fi
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://openrouter.ai/api/v1 VLLM_API_KEY="$KEY" \
    VLLM_MODEL=google/gemma-4-31b-it MODEL_TAG=gemma-4-31B-bf16 \
    LLM_EXTRA_BODY_JSON='{"provider":{"only":["venice"],"allow_fallbacks":false}}' \
    BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=64 SPLIT=test \
    V5=1 CONFIG_ONLY="$ARMS" OUT_TAG=$tag TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag  spent=$(curl -s -m 15 https://openrouter.ai/api/v1/key \
    -H "Authorization: Bearer $KEY" | python3 -c 'import sys,json;print("$%.2f"%json.load(sys.stdin)["data"]["usage"])' 2>/dev/null)"
}
for B in 24000 16000 8000; do run GAIA $B _gor_ga${B%000} GAIA_N=1000; done
for B in 24000 16000 8000; do run MATH $B _gor_ma${B%000} MATH_N=500; done
for B in 24000 16000 8000; do run FRAMES $B _gor_fr${B%000} x=1; done
for B in 24000 16000 8000; do run MUSIQUE $B _gor_mu${B%000} MUSIQUE_N=99999; done
echo "[$(date '+%F %T')] GEMMA BF16 MAIN TABLE DONE"
