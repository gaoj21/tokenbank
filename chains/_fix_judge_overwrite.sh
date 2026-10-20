#!/bin/bash
# Recover from the judge-backup bug: rerun the unsharded 300-task reference repeats on FRAMES/MuSiQue
# (20b now, next to the L1-reason test; 120b after the switch), then judge every FRAMES/MuSiQue file of
# both gpt-oss models with gpt-oss-120b (fixed apply).
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
I=results/climb_live; B="BUDGETS=100000 SPLIT=test L01_MAX_TOKENS=8192"; F="CONFIG_ONLY=V5-Unc-Fmt"
rep() { MODEL=$1 PORT=$2 MAX_WORKERS=128 bash chains/_gptoss_run.sh _gx4kr2_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/repeat300_fr.json &
        MODEL=$1 PORT=$3 MAX_WORKERS=128 bash chains/_gptoss_run.sh _gx4kr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/repeat300_mu.json & }
rep gpt-oss-20b 18121 18122
wait
until grep -qE "L1REASON BOTH DONE" logs/_l1reason_both.log; do sleep 30; done
log "20b work done; switching to 120b"
timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done; for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done; cd /mnt/data/models && (setsid nohup bash serve_gpt_oss_120b.sh > serve_gpt_oss_120b.log 2>&1 < /dev/null &); echo started'
for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-120b && break; sleep 15; done
log "120b ready"
rep gpt-oss-120b 18121 18122
wait
wc -l traces/tasks_sweep_gpt-oss-{20b,120b}_gx4kr2_{fr,mu}100.jsonl
FILES=""; for m in gpt-oss-20b gpt-oss-120b; do for t in climbx_fr100 climbx_mu100 gx4k_fr100 gx4k_mu100 gx4kr2_fr100 gx4kr2_mu100 l1r_fr100 l1r_mu0100; do FILES="$FILES,${m}_$t"; done; done
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_recover.jsonl JUDGE_THREADS=256 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JUDGE_FILES=${FILES#,}
rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { log "JUDGE FAILED"; exit 1; }
log "RECOVER JUDGE DONE"
