#!/bin/bash
# Open-book MuSiQue and FRAMES (OPEN_BOOK=1, _open_book.py) on gpt-oss-20b, served with a 128k context:
# forced climb to L1 (L1_MODE=reason), the reference team from scratch, and its 300-task repeat. No per-task
# budget (the prompts now carry the documents). MuSiQue starts at once; FRAMES waits for the articles.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
tunnels() { for p in "$@"; do ss -ltn | grep -q ":$p " && continue
    setsid nohup ssh -C -i $K -p 820 -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L $p:127.0.0.1:8001 root@47.82.65.107 >/dev/null 2>&1 < /dev/null & sleep 4; done; }
tunnels $(seq 18101 18108) $(seq 18111 18130)
if ! curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q '"openai/gpt-oss-20b".*"max_model_len":131072'; then
    log "switch server -> gpt-oss-20b 128k"
    timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
        for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done
        cd /mnt/data/models && (setsid nohup bash serve_gpt_oss_20b_128k.sh > serve_gpt_oss_20b_128k.log 2>&1 < /dev/null &); echo started'
    for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q '"max_model_len":131072' && break; sleep 15; done
fi
curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q '"max_model_len":131072' || { log "SERVER FAILED"; exit 1; }
log "server ready"
I=results/climb_live; export MODEL=gpt-oss-20b HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
B="BUDGETS=10000000 SPLIT=test L01_MAX_TOKENS=8192 OPEN_BOOK=1"
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 V5_MAX_LEVEL=1 L1_MODE=reason"; F="CONFIG_ONLY=V5-Unc-Fmt"
r() { PORT=$1 MAX_WORKERS=$2 bash chains/_gptoss_run.sh "${@:3}" & }
T=traces/tasks_sweep_$MODEL
i=0; for p in 18111 18112 18113 18114; do r $p 96 _l1robs${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $C TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
i=0; for p in 18115 18116 18117 18118; do r $p 96 _gx4kobs${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
r 18119 96 _gx4kobr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/repeat300_mu.json
until [ -f data/frames_wiki/.complete ]; do sleep 60; done
log "FRAMES articles ready"
r 18120 96 _l1robs0_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half0.json
r 18121 96 _l1robs1_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half1.json
r 18122 96 _gx4kobs0_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/fr_half0.json
r 18123 96 _gx4kobs1_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/fr_half1.json
r 18124 96 _gx4kobr2_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/repeat300_fr.json
wait
for c in l1rob gx4kob; do cat ${T}_${c}s{0,1}_fr100.jsonl > ${T}_${c}_fr100.jsonl; cat ${T}_${c}s{0,1,2,3}_mu100.jsonl > ${T}_${c}_mu100.jsonl; done
wc -l ${T}_{l1rob,gx4kob,gx4kobr2}_{fr,mu}100.jsonl
log "OPENBOOK 20B RUNS DONE"
