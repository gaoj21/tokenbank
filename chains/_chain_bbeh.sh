#!/bin/bash
# BIG-Bench Extra Hard replaces BBH: 874 problems (38 per task), all used. For each model the forced climb
# through L0 and L1 (L1_MODE=reason), the reference team (fixed reviewer) and its 300-task repeat; scored in
# run by _bbeh_judge. gpt-oss-120b, then gpt-oss-20b, then Gemma (local, bf16, eight single-GPU servers).
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
tunnel() { ss -ltn | grep -q ":$1 " && return
    setsid nohup ssh -C -i $K -p 820 -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L $1:127.0.0.1:$2 root@47.82.65.107 >/dev/null 2>&1 < /dev/null & sleep 3; }
killall_vllm() { timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
    for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done; echo gpus_busy=$n'; }
serve_oss() {
    curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q "\"openai/$1\".*\"max_model_len\":131072" && return 0
    killall_vllm
    timeout 120 $SSH "cd /mnt/data/models && (setsid nohup bash serve_${1//-/_}_128k.sh > serve_${1//-/_}_128k.log 2>&1 < /dev/null &); echo started"
    for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q "\"openai/$1\".*\"max_model_len\":131072" && return 0; sleep 15; done
    log "SERVER FAILED $1"; exit 1
}
until grep -q "RERUN ALL DONE\|FAILED" logs/_chain_rerun_ref.log; do sleep 60; done
grep -q "RERUN ALL DONE" logs/_chain_rerun_ref.log || { log "reference rerun chain failed; not starting BBEH"; exit 1; }
for p in $(seq 18101 18108); do tunnel $p 8001; done
URLS=$(for p in $(seq 18101 18108); do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
I=results/climb_live; export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
B="BENCHMARK=BBEH SPLIT=full BUDGETS=10000000 L01_MAX_TOKENS=8192 MAX_TOKENS=16384 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0 LLM_MAX_RETRIES=60"
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 V5_MAX_LEVEL=1 L1_MODE=reason"; F="CONFIG_ONLY=V5-Unc-Fmt"
for m in gpt-oss-120b gpt-oss-20b; do
    serve_oss $m; log "$m server ready"; export MODEL=$m
    PORT=18101 MAX_WORKERS=256 bash chains/_gptoss_run.sh _l1r_bbeh100 $B $C OPENAI_BASE_URLS=$URLS &
    PORT=18101 MAX_WORKERS=256 bash chains/_gptoss_run.sh _gx5k_bbeh100 $B $F OPENAI_BASE_URLS=$URLS &
    PORT=18101 MAX_WORKERS=128 bash chains/_gptoss_run.sh _gx5kr2_bbeh100 $B $F OPENAI_BASE_URLS=$URLS TASK_IDS_FILE=$I/repeat300_bbeh.json &
    wait; wc -l traces/tasks_sweep_${m}_{l1r,gx5k,gx5kr2}_bbeh100.jsonl; log "$m BBEH done"
done
killall_vllm
timeout 120 $SSH 'cd /mnt/data/models; for g in 0 1 2 3 4 5 6 7; do (setsid nohup bash serve_gemma4_26b_a4b_single_128k.sh $g $((8010+g)) > serve_gemma_single128k_$g.log 2>&1 < /dev/null &); sleep 2; done; echo launched'
for i in $(seq 0 7); do tunnel $((18200+i)) $((8010+i)); tunnel $((18210+i)) $((8010+i)); done
for i in $(seq 1 120); do ok=0; for g in 0 1 2 3 4 5 6 7; do curl -s -m 5 http://127.0.0.1:$((18200+g))/v1/models | grep -q '"max_model_len":131072' && ok=$((ok+1)); done; [ $ok = 8 ] && break; sleep 15; done
log "Gemma servers ready: $ok/8"; [ $ok = 8 ] || exit 1
GURLS=$(for p in $(seq 18200 18207) $(seq 18210 18217); do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
GB="$B TOP_LOGPROBS=0 OPENAI_BASE_URLS=$GURLS"
PORT=18200 MAX_WORKERS=384 bash chains/_gemma_local_run.sh _l1r_bbeh100 $GB $C &
PORT=18200 MAX_WORKERS=384 bash chains/_gemma_local_run.sh _gx5k_bbeh100 $GB $F &
PORT=18200 MAX_WORKERS=192 bash chains/_gemma_local_run.sh _gx5kr2_bbeh100 $GB $F TASK_IDS_FILE=$I/repeat300_bbeh.json &
wait; wc -l traces/tasks_sweep_gemma-4-26B-A4B-bf16_{l1r,gx5k,gx5kr2}_bbeh100.jsonl
log "BBEH ALL DONE"
