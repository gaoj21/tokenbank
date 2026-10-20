#!/bin/bash
# LiveCodeBench (844 test problems) on both gpt-oss models: forced climb with the new L1 (single agent,
# full reasoning, from scratch), reference pipeline, and a 300-problem reference repeat; plus the new-L1
# MuSiQue shards 1-3 to complete that comparison. Code calls get a 16,384-token cap (hard problems ran out
# of 4,096 while reasoning). Then judge the new-L1 MuSiQue records with gpt-oss-120b.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
I=results/climb_live
BL="BUDGETS=10000000 SPLIT=test L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason MAX_TOKENS=16384 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0"
BM="BUDGETS=100000 SPLIT=test L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason"
switch() { timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done; for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done; cd /mnt/data/models && (setsid nohup bash '$1' > '${1%.sh}'.log 2>&1 < /dev/null &); echo started'
           for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q "$2" && { log "ready $2"; return 0; }; sleep 15; done; log "SERVER FAILED $2"; exit 1; }
runm() {   # $1 model
    local m=$1; local T=traces/tasks_sweep_$m
    r() { MODEL=$m PORT=$1 MAX_WORKERS=$2 bash chains/_gptoss_run.sh "${@:3}" & }
    local i=0; for p in 18111 18112 18113 18114; do r $p 64 _climbxs${i}_lcb100 BENCHMARK=LCB $BL CONFIG_ONLY=CLIMB-Live-BCB THETA_LIVE=99 TASK_IDS_FILE=$I/lcb_shard$i.json; i=$((i+1)); done
    i=0; for p in 18115 18116 18117 18118; do r $p 64 _gx4ks${i}_lcb100 BENCHMARK=LCB $BL CONFIG_ONLY=BCB-Naive TASK_IDS_FILE=$I/lcb_shard$i.json; i=$((i+1)); done
    r 18119 64 _gx4kr2_lcb100 BENCHMARK=LCB $BL CONFIG_ONLY=BCB-Naive TASK_IDS_FILE=$I/repeat300_lcb.json
    i=1; for p in 18120 18121 18122; do r $p 128 _l1r_mu${i}100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $BM CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
    wait
    for c in climbx gx4k; do cat ${T}_${c}s{0,1,2,3}_lcb100.jsonl > ${T}_${c}_lcb100.jsonl; done
    cat ${T}_l1r_mu{0,1,2,3}100.jsonl > ${T}_l1r_mu100.jsonl
    wc -l ${T}_{climbx,gx4k,gx4kr2}_lcb100.jsonl ${T}_l1r_mu100.jsonl
    log "$m done"
}
curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-120b || switch serve_gpt_oss_120b.sh gpt-oss-120b
runm gpt-oss-120b
switch serve_gpt_oss_20b.sh gpt-oss-20b
runm gpt-oss-20b
switch serve_gpt_oss_120b.sh gpt-oss-120b
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_l1rmu.jsonl JUDGE_THREADS=256 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JUDGE_FILES=gpt-oss-120b_l1r_mu100,gpt-oss-20b_l1r_mu100
rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { log "JUDGE FAILED"; exit 1; }
log "LCB CHAIN DONE"
