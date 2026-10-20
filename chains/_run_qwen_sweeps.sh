#!/bin/bash
# Qwen3.6-27B-FP8 (local vLLM @8001) budget sweeps — cross-model validation of
# the dual-core method (deepseek-v4-flash results already in results/).
# Chain: FRAMES(824) -> MATH(500) -> GAIA(~152), all @ 8K budget, seed 42,
# temp 0, all 6 sweep configs. Each appends to its own OUT_TAG artifacts.
# Requires: bash chains/_start_vllm_conda.sh already serving.
cd "$(dirname "$0")"
MODEL_DIR=$(ls -d /home/j0018/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/ | head -1)
export LLM_BACKEND=vllm
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=Qwen/Qwen3.6-27B-FP8
export TOKENIZER_PATH=$MODEL_DIR
export MODEL_TAG=qwen36-27b-fp8
export TEMPERATURE=0
export SEEDS=42
export BUDGETS=8000
export MAX_WORKERS=64
export HF_HUB_DOWNLOAD_TIMEOUT=30

for SPEC in "FRAMES 824 fr8k" "MATH 500 ma8k" "GAIA 1000 ga8k"; do
  set -- $SPEC
  BENCH=$1; N=$2; TAG=$3
  echo "=== $BENCH n=$N @8K (OUT_TAG=_${TAG}) ==="
  if [ "$BENCH" = "FRAMES" ]; then export FRAMES_N=$N; fi
  if [ "$BENCH" = "MATH" ]; then export MATH_N=$N; fi
  if [ "$BENCH" = "GAIA" ]; then export GAIA_N=$N; fi
  BENCHMARK=$BENCH OUT_TAG=_${TAG} .venv/bin/python -u _budget_sweep_v2.py
done
echo "ALL SWEEPS DONE"
