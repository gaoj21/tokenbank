#!/bin/bash
# v5.1 unified-config dev validation (Gemma chain already done).
cd /home/j0018/Desktop/tokenbank
# anchored pattern: only the real python process, not bash wrappers
while pgrep -f "^\.venv/bin/python -u _budget_sweep" >/dev/null 2>&1; do sleep 60; done
pkill -f "tools/llama.cpp/build/bin/llama-server" 2>/dev/null; sleep 10
bash chains/_start_vllm_conda.sh > logs/_vllm_restart.log 2>&1 &
for i in $(seq 1 40); do
  sleep 15
  if grep -q "Application startup complete" logs/_vllm_restart.log 2>/dev/null; then break; fi
done
sleep 10
run() {
  local bench=$1 tag=$2 log=$3; shift 3
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=24000 SEEDS=42 MAX_WORKERS=64 SPLIT=dev \
  V5=1 CONFIG_ONLY="V5.1-Unified,V5.1-NoPool,V5-A6-OnlyM3,V5-Unconstrained" \
  OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
}
run MATH    _v51_ma _v51_math.log    MATH_N=500
run MUSIQUE _v51_mu _v51_musique.log MUSIQUE_N=99999
run GAIA    _v51_ga _v51_gaia.log    GAIA_N=1000
