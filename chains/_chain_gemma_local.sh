#!/bin/bash
# Gemma-4-31B-it (bf16) on the remote H20s, after the DeepSeek rerun releases the GPUs.
# Same experiment set as the other families: single call (1k/2k), Naive+Fmt at 4096 for all
# three budgets, BCB baseline at 4096, the all-escalate counterfactual at 4096, and a 1024
# deployment control on MATH against the main table's OpenRouter numbers.
cd /home/j0018/Desktop/tokenbank
KEY=/home/j0018/Desktop/remote/id_rsa; RS="ssh -i $KEY -p 820 -o ConnectTimeout=15 root@47.82.65.107"
while ! grep -q "DEEPSEEK RERUN DONE" logs/_chain_rerun4k.log 2>/dev/null; do sleep 60; done
$RS 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
     for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done
     cd /mnt/data/models && setsid nohup bash serve_gemma4_31b.sh > serve_gemma4_31b.log 2>&1 < /dev/null & echo started'
until curl -s -m 5 http://127.0.0.1:18000/v1/models | grep -q gemma-4-31b-it; do sleep 30; done
echo "[$(date '+%F %T')] gemma ready"
TAG=gemma-4-31B-bf16
run() { local t=$1; shift
  [ -s "traces/tasks_sweep_$TAG$t.jsonl" ] && { echo "SKIP $t"; return; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=google/gemma-4-31b-it \
      MODEL_TAG=$TAG SEEDS=42 MAX_WORKERS=512 TEMPERATURE=0 V5=1 OUT_TAG=$t "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
# benchmark tag -> dataset args; tags match the other families (ma, ga, fr, mu)
ma() { echo "BENCHMARK=MATH MATH_N=500 BUDGETS=$1 SPLIT=test"; }
ga() { echo "BENCHMARK=GAIA GAIA_N=1000 BUDGETS=$1 SPLIT=test"; }
fr() { echo "BENCHMARK=FRAMES BUDGETS=$1 SPLIT=test"; }
mu() { echo "BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=$1 SPLIT=test"; }
C() { echo "BENCHMARK=BCB BCB_N=500 BUDGETS=$1 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1"; }
run _gx1k_ma24 $(ma 24000) CONFIG_ONLY=V5-Unc-Fmt
for b in ma ga fr mu; do run _gx_single_${b}24 $($b 24000) CONFIG_ONLY=V5-Single-Fmt-1k,V5-Single-Fmt-2k; done
run _gx_bcbsingle16k $(C 16000) CONFIG_ONLY=BCB-Single-1k,BCB-Single-2k
for K in 24 16 8; do for b in ma ga fr mu; do run _gx4k_${b}${K} $($b ${K}000) MAX_TOKENS=4096 CONFIG_ONLY=V5-Unc-Fmt; done; done
for K in 4 8 16; do run _gx4k_bcb${K}k $(C ${K}000) MAX_TOKENS=4096 CONFIG_ONLY=BCB-Naive; done
for b in ma ga fr mu; do run _gx4ka_${b}24 $($b 24000) MAX_TOKENS=4096 CONFIG_ONLY=V5.1-Ab7-AllEsc; done
echo "[$(date '+%F %T')] GEMMA LOCAL ALL DONE"
