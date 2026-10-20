#!/bin/bash
# Qwen3.8-27B (remote, 4xGPU TP=4 FP8) -- full matrix, queued back-to-back.
# Concurrency 128: measured saturation point (~4,000 tok/s; 192 queues and
# throughput drops). Runs on remote hardware, independent of the local box.
# Order: 24k first (main table), then 16k and 8k (frontier + tightening claim).
cd /home/j0018/Desktop/tokenbank
while pgrep -f "^\.venv/bin/python -u _budget_sweep.*_q38" >/dev/null 2>&1; do sleep 30; done

run() {
  local bench=$1 budget=$2 tag=$3 log=$4; shift 4
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
  VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
  BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=128 SPLIT=test \
  V5=1 CONFIG_ONLY="V5-Unconstrained,V5-Unc-Fmt,V5.1-NoPool" \
  OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}

# --- 24k: the main table ---
run MATH    24000 _q38_ma24 _q38_ma24.log MATH_N=500
run GAIA    24000 _q38_ga24 _q38_ga24.log GAIA_N=1000
run FRAMES  24000 _q38_fr24 _q38_fr24.log
run MUSIQUE 24000 _q38_mu24 _q38_mu24.log MUSIQUE_N=99999
# --- 16k ---
run MATH    16000 _q38_ma16 _q38_ma16.log MATH_N=500
run GAIA    16000 _q38_ga16 _q38_ga16.log GAIA_N=1000
run FRAMES  16000 _q38_fr16 _q38_fr16.log
run MUSIQUE 16000 _q38_mu16 _q38_mu16.log MUSIQUE_N=99999
# --- 8k ---
run MATH    8000  _q38_ma8  _q38_ma8.log  MATH_N=500
run GAIA    8000  _q38_ga8  _q38_ga8.log  GAIA_N=1000
run FRAMES  8000  _q38_fr8  _q38_fr8.log
run MUSIQUE 8000  _q38_mu8  _q38_mu8.log  MUSIQUE_N=99999
echo "[$(date '+%F %T')] Q38 ALL DONE"
