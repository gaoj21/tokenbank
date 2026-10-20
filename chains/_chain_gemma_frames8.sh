#!/bin/bash
# Gemma open-book FRAMES team runs, restarted on all eight servers: the 660-task reference in six shards and
# the 300-task repeat in two (each earlier run was pinned to one server and would have taken ~20 h). Same
# settings as _chain_openbook_gemma.sh. Then wait for the MuSiQue runs, merge, judge, export, compute results.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
I=results/climb_live; export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
B="BUDGETS=10000000 SPLIT=test OPEN_BOOK=1 L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason TOP_LOGPROBS=0 MAX_TOKENS=16384 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0 LLM_MAX_RETRIES=60"
F="CONFIG_ONLY=V5-Unc-Fmt"
g() { PORT=$1 MAX_WORKERS=64 bash chains/_gemma_local_run.sh "${@:2}" $B & }
for i in 0 1 2 3 4 5; do g $((18200+i)) _gx4kobf${i}_fr100 BENCHMARK=FRAMES $F TASK_IDS_FILE=$I/fr_sixth$i.json; done
g 18206 _gx4kobr2f0_fr100 BENCHMARK=FRAMES $F TASK_IDS_FILE=$I/repeat300_fr_half0.json
g 18207 _gx4kobr2f1_fr100 BENCHMARK=FRAMES $F TASK_IDS_FILE=$I/repeat300_fr_half1.json
wait
G=traces/tasks_sweep_gemma-4-26B-A4B-bf16
until [ -s ${G}_gx4kobs0_mu100.jsonl ] && [ -s ${G}_gx4kobs1_mu100.jsonl ] && [ -s ${G}_gx4kobs2_mu100.jsonl ] && [ -s ${G}_gx4kobs3_mu100.jsonl ] && [ -s ${G}_gx4kobr2_mu100.jsonl ]; do sleep 60; done
cat ${G}_gx4kobf{0,1,2,3,4,5}_fr100.jsonl > ${G}_gx4kob_fr100.jsonl
cat ${G}_gx4kobr2f{0,1}_fr100.jsonl > ${G}_gx4kobr2_fr100.jsonl
cat ${G}_l1robs{0,1}_fr100.jsonl > ${G}_l1rob_fr100.jsonl
for c in l1rob gx4kob; do cat ${G}_${c}s{0,1,2,3}_mu100.jsonl > ${G}_${c}_mu100.jsonl; done
wc -l ${G}_{l1rob,gx4kob,gx4kobr2}_{fr,mu}100.jsonl
log "OPENBOOK GEMMA RUNS DONE"
timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
    for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done
    cd /mnt/data/models && (setsid nohup bash serve_gpt_oss_120b_128k.sh > serve_gpt_oss_120b_128k.log 2>&1 < /dev/null &); echo started'
for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-120b && break; sleep 15; done
log "judge server ready"
FILES=""; for t in l1rob_fr100 l1rob_mu100 gx4kob_fr100 gx4kob_mu100 gx4kobr2_fr100 gx4kobr2_mu100; do FILES="$FILES,gemma-4-26B-A4B-bf16_$t"; done
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_obgemma.jsonl JUDGE_THREADS=256 JUDGE_FILES=${FILES#,}
rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { log "JUDGE FAILED"; exit 1; }
RECORDS_OUT=paper/data/records_obgemma.csv SUFFIX=100 .venv/bin/python analysis/export_records_new.py gemma-4-26B-A4B-bf16:Gemma-4-26B-A4B:l1rob:gx4kob
cd paper && GEMMA_TAG=gemma-4-26B-A4B-bf16 CLIMB_TAG=l1rob REPEAT_TAG=gx4kobr2 MAIN_OUT=data/main_results_obgemma.json ../.venv/bin/python analysis/main_results.py data/records_obgemma.csv 2>&1 | grep -v Warn
log "OPENBOOK GEMMA ALL DONE"
