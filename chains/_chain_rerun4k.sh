#!/bin/bash
# Main-table baselines rerun at MAX_TOKENS=4096. The original baselines called the Executor
# with a 1024 cap; on Qwen MATH 31% of items had a call truncated at it and the baseline moved
# from 57.2% to 75.0% when the cap was lifted. 24k was covered by the analysis runs; this adds
# 8k and 16k, and BigCodeBench at all three budgets.
#  Qwen     : remote H20, after the Qwen follow-up chain.
#  DeepSeek : remote H20, after Qwen, swapping the server.
#  GLM      : external SGLang endpoints, after the GLM analysis chain, in parallel with the above.
cd /home/j0018/Desktop/tokenbank
KEY=/home/j0018/Desktop/remote/id_rsa; RS="ssh -i $KEY -p 820 -o ConnectTimeout=15 root@47.82.65.107"
run() { local tag=$1 t=$2; shift 2
  [ -s "traces/tasks_sweep_$tag$t.jsonl" ] && { echo "SKIP $tag$t"; return; }
  echo "[$(date '+%F %T')] START $tag$t"
  env VLLM_API_KEY=EMPTY MODEL_TAG=$tag SEEDS=42 TEMPERATURE=0 V5=1 MAX_TOKENS=4096 OUT_TAG=$t "$@" \
      .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag$t"; }
mainset() {  # $1 tag  $2 prefix  then env
  local tag=$1 pre=$2; shift 2
  for K in 16 8; do
    run $tag ${pre}_ma$K "$@" BENCHMARK=MATH MATH_N=500 BUDGETS=${K}000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt
    run $tag ${pre}_ga$K "$@" BENCHMARK=GAIA GAIA_N=1000 BUDGETS=${K}000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt
    run $tag ${pre}_fr$K "$@" BENCHMARK=FRAMES BUDGETS=${K}000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt
    run $tag ${pre}_mu$K "$@" BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=${K}000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt
  done; }
bcbset() { local tag=$1 pre=$2 arms=$3; shift 3
  for K in 4 8 16; do
    run $tag ${pre}_bcb${K}k "$@" BENCHMARK=BCB BCB_N=500 BUDGETS=${K}000 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1 CONFIG_ONLY=$arms
  done; }
remote_swap() {  # stop whatever vllm serves on the remote, start $1, wait for model id $2
  $RS 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
       for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done'
  $RS "cd /mnt/data/models && setsid nohup bash $1 > ${1%.sh}_rerun.log 2>&1 < /dev/null & echo started $1"
  until curl -s -m 5 http://127.0.0.1:18000/v1/models | grep -q "$2"; do sleep 30; done
  echo "[$(date '+%F %T')] server ready: $2"; }

# ---------- GLM, in background ----------
(
  while ! grep -q "GLM ANALYSIS ALL DONE" logs/_chain_analysis_glm.log 2>/dev/null; do sleep 60; done
  G() { echo "OPENAI_BASE_URL=$1 VLLM_MODEL=$2 MAX_WORKERS=32"; }
  E1=$(G https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8); E2=$(G https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8)
  E3=$(G https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash); E4=$(G https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash)
  E5=$(G https://llm57.jierungogogo.com/v1 zai-org/GLM-5.3-Flash)
  T=GLM-5.3-Flash
  { for K in 16 8; do run $T _glmr4k_ma$K $E1 BENCHMARK=MATH MATH_N=500 BUDGETS=${K}000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt
                      run $T _glmr4k_ga$K $E1 BENCHMARK=GAIA GAIA_N=1000 BUDGETS=${K}000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt; done; } &
  { for K in 16 8; do run $T _glmr4k_fr$K $E2 BENCHMARK=FRAMES BUDGETS=${K}000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt; done; } &
  run $T _glmr4k_mu16 $E3 BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=16000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt &
  run $T _glmr4k_mu8  $E4 BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=8000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt &
  { for K in 4 8; do run $T _glmr4k_bcb${K}k $E5 BENCHMARK=BCB BCB_N=500 BUDGETS=${K}000 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1 CONFIG_ONLY=BCB-Naive,BCB-Ab7; done
    run $T _glmr4k_bcb16k $E5 BENCHMARK=BCB BCB_N=500 BUDGETS=16000 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1 CONFIG_ONLY=BCB-Ab7; } &
  wait; echo "[$(date '+%F %T')] GLM RERUN DONE"
) &

# ---------- Qwen then DeepSeek, on the remote ----------
while ! grep -q "QWEN FOLLOWUP ALL DONE" logs/_chain_analysis_qwen2.log 2>/dev/null; do sleep 60; done
Q="OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_MODEL=qwen3.8-27b-fp8 MAX_WORKERS=512"
mainset Qwen3.8-27B _qr4k $Q
bcbset  Qwen3.8-27B _qr4k BCB-Naive $Q
echo "[$(date '+%F %T')] QWEN RERUN DONE"
remote_swap serve_dsv41.sh deepseek-v4.1-flash
D="OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_MODEL=deepseek-v4.1-flash MAX_WORKERS=256"
mainset DeepSeek-V4.1-Flash _dsr4k $D
bcbset  DeepSeek-V4.1-Flash _dsr4k BCB-Naive,BCB-Ab7 $D
echo "[$(date '+%F %T')] DEEPSEEK RERUN DONE"
wait
echo "[$(date '+%F %T')] RERUN4K ALL DONE"
