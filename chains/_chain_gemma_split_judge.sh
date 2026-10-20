#!/bin/bash
# Gemma open-book: judge and analyse the main runs as soon as they end, while the FRAMES repeat (two shards on
# GPUs 6 and 7) keeps running. Only the Gemma servers on GPUs 0-5 are stopped; gpt-oss-120b judges on those six.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
G=traces/tasks_sweep_gemma-4-26B-A4B-bf16
nonempty() { for f in "$@"; do [ -s $f ] || return 1; done; }
until nonempty ${G}_gx4kobf{0,1,2,3,4,5}_fr100.jsonl ${G}_gx4kobs{0,1,2,3}_mu100.jsonl ${G}_gx4kobr2_mu100.jsonl ${G}_l1robs{0,1}_fr100.jsonl ${G}_l1robs{0,1,2,3}_mu100.jsonl; do sleep 60; done
log "Gemma main runs done"
cat ${G}_gx4kobf{0,1,2,3,4,5}_fr100.jsonl > ${G}_gx4kob_fr100.jsonl
cat ${G}_l1robs{0,1}_fr100.jsonl > ${G}_l1rob_fr100.jsonl
for c in l1rob gx4kob; do cat ${G}_${c}s{0,1,2,3}_mu100.jsonl > ${G}_${c}_mu100.jsonl; done
wc -l ${G}_{l1rob,gx4kob}_{fr,mu}100.jsonl ${G}_gx4kobr2_mu100.jsonl
timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\"" | grep -E -- "--port 801[0-5]( |\$)" | awk "{print \$1}"); do kill $p; done
    for i in $(seq 1 60); do n=$(nvidia-smi -i 0,1,2,3,4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done; echo gpus0to5_busy=$n
    cd /mnt/data/models && (setsid nohup bash serve_gpt_oss_120b_128k_gpu0to5.sh > serve_gpt_oss_120b_128k_gpu0to5.log 2>&1 < /dev/null &); echo started'
for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-120b && break; sleep 15; done
curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-120b || { log "JUDGE SERVER FAILED"; exit 1; }
log "judge server ready on GPUs 0-5"
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl JUDGE_THREADS=256 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
judge() {   # $1 queue name, rest: trace tags
    local FILES=""; for t in "${@:2}"; do FILES="$FILES,gemma-4-26B-A4B-bf16_$t"; done
    export JUDGE_QUEUE=results/judge_queue_oss120b_$1.jsonl JUDGE_FILES=${FILES#,}; rm -f $JUDGE_QUEUE
    .venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply
}
judge obgemma l1rob_fr100 l1rob_mu100 gx4kob_fr100 gx4kob_mu100 gx4kobr2_mu100 || { log "JUDGE FAILED"; exit 1; }
analyse() {
    RECORDS_OUT=paper/data/records_obgemma.csv SUFFIX=100 .venv/bin/python analysis/export_records_new.py gemma-4-26B-A4B-bf16:Gemma-4-26B-A4B:l1rob:gx4kob
    .venv/bin/python - <<'PY'
import pandas as pd
d = pd.read_csv('paper/data/records_final.csv'); g = pd.read_csv('paper/data/records_obgemma.csv')
d = d[~((d.model == 'Gemma-4-26B-A4B') & d.bench.isin(['FRAMES', 'MuSiQue']))]
pd.concat([d, g]).to_csv('paper/data/records_final.csv', index=False); print(pd.concat([d, g]).groupby(['model', 'bench']).size().to_dict())
PY
    bash chains/_rerun_analyses_ob.sh
}
analyse
log "GEMMA MAIN DONE"
until nonempty ${G}_gx4kobr2f0_fr100.jsonl ${G}_gx4kobr2f1_fr100.jsonl; do sleep 60; done
cat ${G}_gx4kobr2f{0,1}_fr100.jsonl > ${G}_gx4kobr2_fr100.jsonl; wc -l ${G}_gx4kobr2_fr100.jsonl
judge obgemma_r2fr gx4kobr2_fr100 || { log "REPEAT JUDGE FAILED"; exit 1; }
(cd paper && ../.venv/bin/python analysis/main_results.py data/records_final.csv 2>&1 | grep -v Warn && ../.venv/bin/python analysis/make_tab_main.py && ../.venv/bin/python analysis/appendix_extras.py 2>&1 | tail -1)
log "GEMMA ALL DONE"
