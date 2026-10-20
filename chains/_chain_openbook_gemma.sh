#!/bin/bash
# After the gpt-oss-120b open-book chain: Gemma-4-26B-A4B (standard checkpoint, bf16, eight single-GPU servers,
# 128k context) on open-book MuSiQue and FRAMES with the Makora run's settings; then gpt-oss-120b judges,
# records are exported and the results computed.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
tunnel() { ss -ltn | grep -q ":$1 " && return
    setsid nohup ssh -C -i $K -p 820 -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L $1:127.0.0.1:$2 root@47.82.65.107 >/dev/null 2>&1 < /dev/null & sleep 3; }
killall_vllm() { timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
    for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done; echo gpus_busy=$n'; }
until grep -q "OPENBOOK 120B ALL DONE\|JUDGE FAILED\|SERVER FAILED" logs/_chain_openbook_120b.log; do sleep 60; done
grep -q "OPENBOOK 120B ALL DONE" logs/_chain_openbook_120b.log || { log "120b chain failed, not starting Gemma"; exit 1; }
log "starting Gemma servers"
killall_vllm
timeout 120 $SSH 'cd /mnt/data/models; for g in 0 1 2 3 4 5 6 7; do (setsid nohup bash serve_gemma4_26b_a4b_single_128k.sh $g $((8010+g)) > serve_gemma_single128k_$g.log 2>&1 < /dev/null &); sleep 2; done; echo launched'
for i in $(seq 0 15); do tunnel $((18200+i)) $((8010+i%8)); done
for i in $(seq 1 120); do ok=0; for g in 0 1 2 3 4 5 6 7; do curl -s -m 5 http://127.0.0.1:$((18200+g))/v1/models | grep -q '"max_model_len":131072' && ok=$((ok+1)); done; [ $ok = 8 ] && break; sleep 15; done
log "Gemma servers ready: $ok/8"; [ $ok = 8 ] || exit 1
I=results/climb_live; export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
B="BUDGETS=10000000 SPLIT=test OPEN_BOOK=1 L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason TOP_LOGPROBS=0 MAX_TOKENS=16384 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0"
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"; F="CONFIG_ONLY=V5-Unc-Fmt"
g() { PORT=$1 MAX_WORKERS=64 bash chains/_gemma_local_run.sh "${@:2}" $B & }
# the heavy FRAMES team runs get a server each; the rest share
g 18200 _gx4kobs0_fr100 BENCHMARK=FRAMES $F TASK_IDS_FILE=$I/fr_half0.json
g 18201 _gx4kobs1_fr100 BENCHMARK=FRAMES $F TASK_IDS_FILE=$I/fr_half1.json
g 18202 _gx4kobr2_fr100 BENCHMARK=FRAMES $F TASK_IDS_FILE=$I/repeat300_fr.json
g 18203 _l1robs0_fr100 BENCHMARK=FRAMES $C TASK_IDS_FILE=$I/fr_half0.json
g 18211 _l1robs1_fr100 BENCHMARK=FRAMES $C TASK_IDS_FILE=$I/fr_half1.json
i=0; for p in 18204 18205 18206 18207; do g $p _gx4kobs${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $F TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
g 18212 _gx4kobr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $F TASK_IDS_FILE=$I/repeat300_mu.json
i=0; for p in 18213 18214 18215 18208; do g $p _l1robs${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $C TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
wait
G=traces/tasks_sweep_gemma-4-26B-A4B-bf16
for c in l1rob gx4kob; do cat ${G}_${c}s{0,1}_fr100.jsonl > ${G}_${c}_fr100.jsonl; cat ${G}_${c}s{0,1,2,3}_mu100.jsonl > ${G}_${c}_mu100.jsonl; done
wc -l ${G}_{l1rob,gx4kob,gx4kobr2}_{fr,mu}100.jsonl
log "OPENBOOK GEMMA RUNS DONE"
killall_vllm
timeout 120 $SSH 'cd /mnt/data/models && (setsid nohup bash serve_gpt_oss_120b_128k.sh > serve_gpt_oss_120b_128k.log 2>&1 < /dev/null &); echo started'
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
