#!/bin/bash
# BBH (999 problems: 37 per task over 27 tasks, all used) on both gpt-oss models: forced climb with the new
# L1, reference pipeline and a 300-problem reference repeat. Scored in-run by the official-style match.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
I=results/climb_live
B="BUDGETS=10000000 SPLIT=full L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason MAX_TOKENS=16384 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0"
switch() { timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done; for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done; cd /mnt/data/models && (setsid nohup bash '$1' > '${1%.sh}'.log 2>&1 < /dev/null &); echo started'
           for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q "$2" && { log "ready $2"; return 0; }; sleep 15; done; log "SERVER FAILED $2"; exit 1; }
bbh() {
    local m=$1; local T=traces/tasks_sweep_$m
    r() { MODEL=$m PORT=$1 MAX_WORKERS=$2 bash chains/_gptoss_run.sh "${@:3}" & }
    local i=0; for p in 18111 18112 18113 18114; do r $p 128 _climbxs${i}_bbh100 BENCHMARK=BBH $B CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 TASK_IDS_FILE=$I/bbh_shard$i.json; i=$((i+1)); done
    i=0; for p in 18115 18116 18117 18118; do r $p 128 _gx4ks${i}_bbh100 BENCHMARK=BBH $B CONFIG_ONLY=V5-Unc-Fmt TASK_IDS_FILE=$I/bbh_shard$i.json; i=$((i+1)); done
    r 18119 128 _gx4kr2_bbh100 BENCHMARK=BBH $B CONFIG_ONLY=V5-Unc-Fmt TASK_IDS_FILE=$I/repeat300_bbh.json
    wait
    for c in climbx gx4k; do cat ${T}_${c}s{0,1,2,3}_bbh100.jsonl > ${T}_${c}_bbh100.jsonl; done
    wc -l ${T}_{climbx,gx4k,gx4kr2}_bbh100.jsonl; log "$m BBH done"
}
curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-120b || switch serve_gpt_oss_120b.sh gpt-oss-120b
bbh gpt-oss-120b
switch serve_gpt_oss_20b.sh gpt-oss-20b
bbh gpt-oss-20b
log "BBH CHAIN DONE"
