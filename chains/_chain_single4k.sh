#!/bin/bash
# Single call at the 4096 per-call cap, all three families, so single-call vs pipeline is
# compared at an equal cap. Qwen runs now beside the jobs already on the H20 server; DeepSeek
# waits for its server in the rerun chain; GLM uses llm3 alongside the GLM chains.
cd /home/j0018/Desktop/tokenbank
run() { local tag=$1 t=$2; shift 2
  [ -s "traces/tasks_sweep_$tag$t.jsonl" ] && { echo "SKIP $tag$t"; return; }
  echo "[$(date '+%F %T')] START $tag$t"
  env VLLM_API_KEY=EMPTY MODEL_TAG=$tag SEEDS=42 TEMPERATURE=0 V5=1 OUT_TAG=$t "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag$t"; }
set5() { local tag=$1 pre=$2; shift 2
  run $tag ${pre}_ma24 "$@" BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test CONFIG_ONLY=V5-Single-Fmt-4k
  run $tag ${pre}_ga24 "$@" BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SPLIT=test CONFIG_ONLY=V5-Single-Fmt-4k
  run $tag ${pre}_fr24 "$@" BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test CONFIG_ONLY=V5-Single-Fmt-4k
  run $tag ${pre}_mu24 "$@" BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test CONFIG_ONLY=V5-Single-Fmt-4k
  run $tag ${pre}_bcb16k "$@" BENCHMARK=BCB BCB_N=500 BUDGETS=16000 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1 CONFIG_ONLY=BCB-Single-4k; }
set5 GLM-5.3-Flash _glms4k OPENAI_BASE_URL=https://llm3.ruoyu-wang.com/v1 VLLM_MODEL=GLM-5.3-Flash-FP8 MAX_WORKERS=32 &
if curl -s -m 5 http://127.0.0.1:18000/v1/models | grep -q qwen3.8-27b-fp8; then
  set5 Qwen3.8-27B _qs4k OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_MODEL=qwen3.8-27b-fp8 MAX_WORKERS=256
fi
until grep -q "server ready: deepseek" logs/_chain_rerun4k.log 2>/dev/null; do sleep 60; done
set5 DeepSeek-V4.1-Flash _dss4k OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_MODEL=deepseek-v4.1-flash MAX_WORKERS=128
wait; echo "[$(date '+%F %T')] SINGLE4K ALL DONE"
