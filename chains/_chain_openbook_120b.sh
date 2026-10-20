#!/bin/bash
# Open-book MuSiQue and FRAMES on gpt-oss-120b (128k context): forced climb to L1 (L1_MODE=reason), the
# reference team, its 300-task repeat; then judge on the same server (gpt-oss-120b is the judge), export, results.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
tunnels() { for p in "$@"; do ss -ltn | grep -q ":$p " && continue
    setsid nohup ssh -C -i $K -p 820 -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L $p:127.0.0.1:8001 root@47.82.65.107 >/dev/null 2>&1 < /dev/null & sleep 4; done; }
tunnels $(seq 18101 18108) $(seq 18111 18130)
log "switch server -> gpt-oss-120b 128k"
timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
    for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done
    cd /mnt/data/models && (setsid nohup bash serve_gpt_oss_120b_128k.sh > serve_gpt_oss_120b_128k.log 2>&1 < /dev/null &); echo started'
for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q '"openai/gpt-oss-120b".*"max_model_len":131072' && break; sleep 15; done
curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q '"max_model_len":131072' || { log "SERVER FAILED"; exit 1; }
log "server ready"
I=results/climb_live; export MODEL=gpt-oss-120b HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
B="BUDGETS=10000000 SPLIT=test L01_MAX_TOKENS=8192 OPEN_BOOK=1"
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 V5_MAX_LEVEL=1 L1_MODE=reason"; F="CONFIG_ONLY=V5-Unc-Fmt"
r() { PORT=$1 MAX_WORKERS=$2 bash chains/_gptoss_run.sh "${@:3}" & }
T=traces/tasks_sweep_$MODEL
i=0; for p in 18111 18112 18113 18114; do r $p 96 _l1robs${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $C TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
i=0; for p in 18115 18116 18117 18118; do r $p 96 _gx4kobs${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
r 18119 96 _gx4kobr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/repeat300_mu.json
r 18120 96 _l1robs0_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half0.json
r 18121 96 _l1robs1_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half1.json
r 18122 96 _gx4kobs0_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/fr_half0.json
r 18123 96 _gx4kobs1_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/fr_half1.json
r 18124 96 _gx4kobr2_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/repeat300_fr.json
wait
for c in l1rob gx4kob; do cat ${T}_${c}s{0,1}_fr100.jsonl > ${T}_${c}_fr100.jsonl; cat ${T}_${c}s{0,1,2,3}_mu100.jsonl > ${T}_${c}_mu100.jsonl; done
wc -l ${T}_{l1rob,gx4kob,gx4kobr2}_{fr,mu}100.jsonl
log "OPENBOOK 120B RUNS DONE"
FILES=""; for t in l1rob_fr100 l1rob_mu100 gx4kob_fr100 gx4kob_mu100 gx4kobr2_fr100 gx4kobr2_mu100; do FILES="$FILES,gpt-oss-120b_$t"; done
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_ob120b.jsonl JUDGE_THREADS=256 JUDGE_FILES=${FILES#,}
rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { log "JUDGE FAILED"; exit 1; }
RECORDS_OUT=paper/data/records_ob120b.csv SUFFIX=100 .venv/bin/python analysis/export_records_new.py gpt-oss-120b:gpt-oss-120b:l1rob:gx4kob
cd paper && CLIMB_TAG=l1rob REPEAT_TAG=gx4kobr2 MAIN_OUT=data/main_results_ob120b.json ../.venv/bin/python analysis/main_results.py data/records_ob120b.csv 2>&1 | grep -v Warn
log "OPENBOOK 120B ALL DONE"
