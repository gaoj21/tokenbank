#!/bin/bash
# Judge the gpt-oss-20b open-book FRAMES/MuSiQue runs with gpt-oss-120b (medium, temperature 0), then export records.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
    for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done
    cd /mnt/data/models && (setsid nohup bash serve_gpt_oss_120b.sh > serve_gpt_oss_120b.log 2>&1 < /dev/null &); echo started'
for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-120b && break; sleep 15; done
log "judge server ready"
FILES=""; for t in l1rob_fr100 l1rob_mu100 gx4kob_fr100 gx4kob_mu100 gx4kobr2_fr100 gx4kobr2_mu100; do FILES="$FILES,gpt-oss-20b_$t"; done
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_ob20b.jsonl JUDGE_THREADS=256 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JUDGE_FILES=${FILES#,}
rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { log "JUDGE FAILED"; exit 1; }
RECORDS_OUT=paper/data/records_ob20b.csv SUFFIX=100 .venv/bin/python analysis/export_records_new.py gpt-oss-20b:gpt-oss-20b:l1rob:gx4kob
log "JUDGE OB20B DONE"
