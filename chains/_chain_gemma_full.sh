#!/bin/bash
# Gemma-4-26B-A4B (Google QAT checkpoint, FP8 online weights, eight single-GPU servers), full task sets as for
# gpt-oss: MATH 400, FRAMES 660, MuSiQue 1934, BBH 999; forced climb with the new L1 (single agent, thinking on,
# from scratch) stopping after L1, reference pipeline, 300-task reference repeats. No per-task budget,
# 16,384-token calls, cheap rungs capped at 8,192, 1 h request timeout. Then judge FRAMES/MuSiQue with
# gpt-oss-120b and rescore MATH from the stored replies.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
I=results/climb_live; S=/tmp/claude-1001/-home-j0018-Desktop-tokenbank/f14fef7e-adf9-443b-9307-5c07475168b4/scratchpad
B="BUDGETS=10000000 L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason TOP_LOGPROBS=0 MAX_TOKENS=16384 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0"
killall_vllm() { timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done; for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done; echo gpus_busy=$n'; }
tunnel() { ss -ltn | grep -q ":$1 " || { setsid nohup ssh -C -i $K -p 820 -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L $1:127.0.0.1:$2 root@47.82.65.107 >/dev/null 2>&1 < /dev/null & sleep 4; }; }

until grep -qE "BBH CHAIN DONE|SERVER FAILED" logs/_chain_bbh2.log; do sleep 60; done
log "gpt-oss BBH finished; starting Gemma"
killall_vllm
timeout 120 $SSH 'cd /mnt/data/models; for g in 0 1 2 3 4 5 6 7; do (setsid nohup bash serve_gemma4_26b_qat_fp8_single.sh $g $((8010+g)) > serve_gemma_qat_$g.log 2>&1 < /dev/null &); sleep 2; done; echo launched'
for i in $(seq 0 31); do tunnel $((18200+i)) $((8010+i%8)); done
for i in $(seq 1 80); do ok=0; for g in 0 1 2 3 4 5 6 7; do curl -s -m 5 http://127.0.0.1:$((18200+g))/v1/models | grep -q a4b-it-qat && ok=$((ok+1)); done; [ $ok = 8 ] && break; sleep 15; done
log "QAT servers ready: $ok/8"; [ $ok = 8 ] || exit 1

export GEMMA_MODEL=google/gemma-4-26b-a4b-it-qat GEMMA_TAG=gemma-4-26B-A4B-qat
G=traces/tasks_sweep_gemma-4-26B-A4B-qat
mkdir -p _archive/traces_gemma_qat_old && mv ${G}_* logs/gemma-4-26B-A4B-qat_* _archive/traces_gemma_qat_old/ 2>/dev/null
R=chains/_gemma_run.sh; C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"; F="CONFIG_ONLY=V5-Unc-Fmt"
P=18200; grun() { PORT=$P MAX_WORKERS=$1 bash $R "${@:2}" $B & P=$((P+1)); }
for t in l1r gx4k; do
    [ $t = l1r ] && X="$C" || X="$F"
    grun 96 _${t}_ma100 BENCHMARK=MATH MATH_N=500 SPLIT=test $X
    grun 96 _${t}s0_fr100 BENCHMARK=FRAMES SPLIT=test $X TASK_IDS_FILE=$I/fr_half0.json
    grun 96 _${t}s1_fr100 BENCHMARK=FRAMES SPLIT=test $X TASK_IDS_FILE=$I/fr_half1.json
    for i in 0 1 2 3; do grun 96 _${t}s${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 SPLIT=test $X TASK_IDS_FILE=$I/oss_mu_shard$i.json; done
    for i in 0 1 2 3; do grun 96 _${t}s${i}_bbh100 BENCHMARK=BBH SPLIT=full $X TASK_IDS_FILE=$I/bbh_shard$i.json; done
done
grun 96 _gx4kr2_ma100 BENCHMARK=MATH MATH_N=500 SPLIT=test $F TASK_IDS_FILE=$I/repeat300_ma.json
grun 96 _gx4kr2_fr100 BENCHMARK=FRAMES SPLIT=test $F TASK_IDS_FILE=$I/repeat300_fr.json
grun 96 _gx4kr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 SPLIT=test $F TASK_IDS_FILE=$I/repeat300_mu.json
grun 96 _gx4kr2_bbh100 BENCHMARK=BBH SPLIT=full $F TASK_IDS_FILE=$I/repeat300_bbh.json
wait
for t in l1r gx4k; do
    cat ${G}_${t}s{0,1}_fr100.jsonl > ${G}_${t}_fr100.jsonl
    cat ${G}_${t}s{0,1,2,3}_mu100.jsonl > ${G}_${t}_mu100.jsonl
    cat ${G}_${t}s{0,1,2,3}_bbh100.jsonl > ${G}_${t}_bbh100.jsonl
done
wc -l ${G}_{l1r,gx4k}_{ma,fr,mu,bbh}100.jsonl ${G}_gx4kr2_{ma,fr,mu,bbh}100.jsonl
log "Gemma runs done"

killall_vllm
timeout 120 $SSH 'cd /mnt/data/models && (setsid nohup bash serve_gpt_oss_120b.sh > serve_gpt_oss_120b.log 2>&1 < /dev/null &); echo started'
for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-120b && break; sleep 15; done
log "judge server ready"
FILES=""; for t in l1r_fr100 l1r_mu100 gx4k_fr100 gx4k_mu100 gx4kr2_fr100 gx4kr2_mu100; do FILES="$FILES,gemma-4-26B-A4B-qat_$t"; done
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_gemma_full.jsonl JUDGE_THREADS=256 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JUDGE_FILES=${FILES#,}
rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { log "JUDGE FAILED"; exit 1; }
sed "s#glob.glob('traces/tasks_sweep_\*.jsonl')#glob.glob('traces/tasks_sweep_gemma-4-26B-A4B-qat_*_ma100.jsonl')#" analysis/rescore_math.py > $S/rescore_gemma_full.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python $S/rescore_gemma_full.py | tail -1
log "GEMMA FULL DONE"
