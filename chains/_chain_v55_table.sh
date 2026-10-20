#!/bin/bash
# v5.5: fill the empty cells of the main table (Table 1).
# Missing: Unc@8k x4, Unc+Fmt@8k x2, Unc+Fmt@16k x MuSiQue, Ours@8k x2.
# Ordered cheapest-first so the table fills progressively.
set -e
cd /home/j0018/Desktop/tokenbank

# --- swap llama-server (Gemma) -> vLLM (Qwen) ---
pkill -f "llama-server" || true
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "$used" -lt 2000 ] && break
  sleep 10
done
echo "GPU drained: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ')"
nohup bash chains/_start_vllm_conda.sh > logs/_vllm_v55.log 2>&1 &
for i in $(seq 1 120); do
  curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q Qwen && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8001/v1/models | head -c 120; echo

run() {
  local bench=$1 budget=$2 cfgs=$3 tag=$4 log=$5; shift 5
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=64 SPLIT=test \
  V5=1 CONFIG_ONLY="$cfgs" OUT_TAG=$tag TEMPERATURE=0 "$@" \
  .venv/bin/python -u _budget_sweep_v2.py > logs/$log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}

ALL="V5-Unconstrained,V5-Unc-Fmt,V5.1-NoPool"
run GAIA    8000  "V5-Unconstrained" _v55_ga8_unc  _v55_ga8.log  GAIA_N=1000
run MATH    8000  "$ALL"             _v55_ma8      _v55_ma8.log  MATH_N=500
run FRAMES  8000  "V5-Unconstrained" _v55_fr8_unc  _v55_fr8.log
run MUSIQUE 16000 "V5-Unc-Fmt"       _v55_mu16_fmt _v55_mu16.log MUSIQUE_N=99999
run MUSIQUE 8000  "$ALL"             _v55_mu8      _v55_mu8.log  MUSIQUE_N=99999
echo "[$(date '+%F %T')] ALL DONE"
