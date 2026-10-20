#!/bin/bash
# Rerun every reference-team run (and its 300-task repeat) with the fixed reviewer verdict (VERDICT line,
# wording fallback, closing call for empty reviews). The climb runs (L0, L1) have no reviewer and are kept.
# New tags: gx5k / gx5kr2 (MATH, BBH), gx5kob / gx5kobr2 (open-book FRAMES, MuSiQue).
# Order: gpt-oss-20b, gpt-oss-120b, judge both (gpt-oss-120b) and rescore MATH, then Gemma FRAMES/MuSiQue locally.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
tunnel() { ss -ltn | grep -q ":$1 " && return
    setsid nohup ssh -C -i $K -p 820 -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L $1:127.0.0.1:$2 root@47.82.65.107 >/dev/null 2>&1 < /dev/null & sleep 3; }
killall_vllm() { timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
    for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done; echo gpus_busy=$n'; }
serve_oss() {   # $1 model (gpt-oss-20b|gpt-oss-120b)
    killall_vllm
    timeout 120 $SSH "cd /mnt/data/models && (setsid nohup bash serve_${1//-/_}_128k.sh > serve_${1//-/_}_128k.log 2>&1 < /dev/null &); echo started"
    for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q "\"openai/$1\".*\"max_model_len\":131072" && return 0; sleep 15; done
    log "SERVER FAILED $1"; exit 1
}
for p in $(seq 18101 18108); do tunnel $p 8001; done
URLS=$(for p in $(seq 18101 18108); do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
I=results/climb_live; export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
R="CONFIG_ONLY=V5-Unc-Fmt LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0 LLM_MAX_RETRIES=60 OPENAI_BASE_URLS=$URLS"
oss_runs() {   # $1 model
    export MODEL=$1
    r() { PORT=18101 MAX_WORKERS=$1 bash chains/_gptoss_run.sh "${@:2}" $R & }
    r 128 _gx5k_ma100 BENCHMARK=MATH MATH_N=500 BUDGETS=100000 SPLIT=test
    r 96 _gx5kr2_ma100 BENCHMARK=MATH MATH_N=500 BUDGETS=100000 SPLIT=test TASK_IDS_FILE=$I/repeat300_ma.json
    r 192 _gx5k_bbh100 BENCHMARK=BBH BUDGETS=10000000 SPLIT=full MAX_TOKENS=16384
    r 96 _gx5kr2_bbh100 BENCHMARK=BBH BUDGETS=10000000 SPLIT=full MAX_TOKENS=16384 TASK_IDS_FILE=$I/repeat300_bbh.json
    r 160 _gx5kob_fr100 BENCHMARK=FRAMES BUDGETS=10000000 SPLIT=test OPEN_BOOK=1 TASK_IDS_FILE=$I/fr_all.json
    r 96 _gx5kobr2_fr100 BENCHMARK=FRAMES BUDGETS=10000000 SPLIT=test OPEN_BOOK=1 TASK_IDS_FILE=$I/repeat300_fr.json
    r 256 _gx5kob_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=10000000 SPLIT=test OPEN_BOOK=1 TASK_IDS_FILE=$I/mu_all.json
    r 96 _gx5kobr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=10000000 SPLIT=test OPEN_BOOK=1 TASK_IDS_FILE=$I/repeat300_mu.json
    wait
    wc -l traces/tasks_sweep_${1}_gx5k*_*100.jsonl
    log "$1 reference reruns done"
}
serve_oss gpt-oss-20b; log "20b server ready"; oss_runs gpt-oss-20b
serve_oss gpt-oss-120b; log "120b server ready"; oss_runs gpt-oss-120b
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$URLS
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl JUDGE_THREADS=256
FILES=""; for m in gpt-oss-20b gpt-oss-120b; do for t in gx5kob_fr100 gx5kob_mu100 gx5kobr2_fr100 gx5kobr2_mu100; do FILES="$FILES,${m}_$t"; done; done
export JUDGE_QUEUE=results/judge_queue_oss120b_gx5k_oss.jsonl JUDGE_FILES=${FILES#,}; rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { log "OSS JUDGE FAILED"; exit 1; }
S=/tmp/claude-1001/-home-j0018-Desktop-tokenbank/f14fef7e-adf9-443b-9307-5c07475168b4/scratchpad
sed "s#glob.glob('traces/tasks_sweep_\*.jsonl')#glob.glob('traces/tasks_sweep_gpt-oss-*_gx5k*_ma100.jsonl')#" analysis/rescore_math.py > $S/rescore_gx5k_oss.py
.venv/bin/python $S/rescore_gx5k_oss.py | tail -2
log "OSS ALL DONE"
# Gemma FRAMES/MuSiQue locally: eight single-GPU servers, every run spread over all eight.
killall_vllm
timeout 120 $SSH 'cd /mnt/data/models; for g in 0 1 2 3 4 5 6 7; do (setsid nohup bash serve_gemma4_26b_a4b_single_128k.sh $g $((8010+g)) > serve_gemma_single128k_$g.log 2>&1 < /dev/null &); sleep 2; done; echo launched'
for i in $(seq 0 7); do tunnel $((18200+i)) $((8010+i)); tunnel $((18210+i)) $((8010+i)); done
for i in $(seq 1 120); do ok=0; for g in 0 1 2 3 4 5 6 7; do curl -s -m 5 http://127.0.0.1:$((18200+g))/v1/models | grep -q '"max_model_len":131072' && ok=$((ok+1)); done; [ $ok = 8 ] && break; sleep 15; done
log "Gemma servers ready: $ok/8"; [ $ok = 8 ] || exit 1
GURLS=$(for p in $(seq 18200 18207) $(seq 18210 18217); do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
GB="BUDGETS=10000000 SPLIT=test OPEN_BOOK=1 TOP_LOGPROBS=0 MAX_TOKENS=16384 CONFIG_ONLY=V5-Unc-Fmt LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0 LLM_MAX_RETRIES=60 OPENAI_BASE_URLS=$GURLS"
g() { PORT=18200 MAX_WORKERS=$1 bash chains/_gemma_local_run.sh "${@:2}" $GB & }
g 384 _gx5kob_fr100 BENCHMARK=FRAMES TASK_IDS_FILE=$I/fr_all.json
g 192 _gx5kobr2_fr100 BENCHMARK=FRAMES TASK_IDS_FILE=$I/repeat300_fr.json
g 512 _gx5kob_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 TASK_IDS_FILE=$I/mu_all.json
g 192 _gx5kobr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 TASK_IDS_FILE=$I/repeat300_mu.json
wait
G=traces/tasks_sweep_gemma-4-26B-A4B-bf16; wc -l ${G}_gx5kob*_100.jsonl ${G}_gx5kob*_fr100.jsonl ${G}_gx5kob*_mu100.jsonl 2>/dev/null
log "GEMMA OB RERUN DONE"
killall_vllm; serve_oss gpt-oss-120b
FILES=""; for t in gx5kob_fr100 gx5kob_mu100 gx5kobr2_fr100 gx5kobr2_mu100; do FILES="$FILES,gemma-4-26B-A4B-bf16_$t"; done
export JUDGE_QUEUE=results/judge_queue_oss120b_gx5k_gemma.jsonl JUDGE_FILES=${FILES#,}; rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { log "GEMMA JUDGE FAILED"; exit 1; }
log "RERUN ALL DONE"
