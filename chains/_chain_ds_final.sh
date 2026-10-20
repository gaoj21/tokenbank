#!/bin/bash
# Two-family paper (GLM + DeepSeek). Qwen dropped 2026-10-02. Swap the remote server to
# DeepSeek and run what DeepSeek still lacks: Naive+Fmt at cap 4096 for 8k and 16k budgets,
# BigCodeBench baselines at cap 4096, and the single call at cap 4096.
# SEGMENT_ACCOUNTING=1 throughout: it only records where prompt characters come from, and the
# 16k Naive+Fmt cells double as the budget accounting of Section 4.
cd /home/j0018/Desktop/tokenbank
RS="ssh -i /home/j0018/Desktop/remote/id_rsa -p 820 -o ConnectTimeout=15 root@47.82.65.107"
# server swap already issued by the first launch of this chain; just wait for it
until curl -s -m 5 http://127.0.0.1:18000/v1/models | grep -q deepseek-v4.1-flash; do sleep 30; done
echo "[$(date '+%F %T')] server ready: deepseek"
T=DeepSeek-V4.1-Flash
run() { local t=$1; shift
  [ -s "traces/tasks_sweep_$T$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=deepseek-v4.1-flash MODEL_TAG=$T \
      SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=256 SEGMENT_ACCOUNTING=1 OUT_TAG=$t "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
ma() { echo "BENCHMARK=MATH MATH_N=500 BUDGETS=$1 SPLIT=test"; }
ga() { echo "BENCHMARK=GAIA GAIA_N=1000 BUDGETS=$1 SPLIT=test"; }
fr() { echo "BENCHMARK=FRAMES BUDGETS=$1 SPLIT=test"; }
mu() { echo "BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=$1 SPLIT=test"; }
bc() { echo "BENCHMARK=BCB BCB_N=500 BUDGETS=$1 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1"; }
# single call at 4096 first: cheap, and it is the comparison the paper rests on
for b in ma ga fr mu; do run _dss4k_${b}24 $($b 24000) CONFIG_ONLY=V5-Single-Fmt-4k; done
run _dss4k_bcb16k $(bc 16000) CONFIG_ONLY=BCB-Single-4k
for K in 16 8; do for b in ma ga fr mu; do run _dsr4k_${b}$K $($b ${K}000) MAX_TOKENS=4096 CONFIG_ONLY=V5-Unc-Fmt; done; done
for K in 4 8 16; do run _dsr4k_bcb${K}k $(bc ${K}000) MAX_TOKENS=4096 CONFIG_ONLY=BCB-Naive,BCB-Ab7; done
echo "[$(date '+%F %T')] DS FINAL ALL DONE"
