#!/bin/bash
# Overnight relay, part 7: Gemma (FP8 weights) on MuSiQue shards 0-1 only and without repeats, to finish by morning. The cheap rungs on BigCodeBench now ask for the complete
# function after running their code (gpt-oss-120b answered in prose half the time), so every model's
# forced-climb BigCodeBench run is redone with that prompt; reference runs are unaffected.
#   C. Gemma-4-26B-A4B: restart the BCB climb shards, wait for the main runs, then the repeats; merge.
#   D. gpt-oss-120b: judge FRAMES/MuSiQue (20b, Gemma, 120b repeats) while rerunning its BCB climb.
#   E. gpt-oss-20b: rerun its BCB climb. Then rescore MATH and export all records.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
I=results/climb_live; B="BUDGETS=100000 SPLIT=test"; S=/tmp/claude-1001/-home-j0018-Desktop-tokenbank/f14fef7e-adf9-443b-9307-5c07475168b4/scratchpad
log() { echo "[$(date '+%F %T')] $*"; }
tunnels() {
    local rp=$1; shift
    for p in "$@"; do
        ss -ltn | grep -q ":$p " && continue
        setsid nohup ssh -C -i $K -p 820 -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
            -N -L $p:127.0.0.1:$rp root@47.82.65.107 >/dev/null 2>&1 < /dev/null &
        sleep 4
    done
}
P8001="$(seq 18101 18108) $(seq 18111 18130)"; P8000="$(seq 18000 18008) $(seq 18010 18029)"
switch() {
    log "switch server -> $1"
    timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
        for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done
        cd /mnt/data/models && (setsid nohup bash '$1' > '${1%.sh}'.log 2>&1 < /dev/null &); echo started'
    for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:$2/v1/models | grep -q "$3" && { log "server ready: $3"; return 0; }; sleep 15; done
    log "SERVER FAILED: $3"; return 1
}
gemma_alive() { for p in $(pgrep -f _budget_sweep_v2.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^MODEL_TAG=gemma-4-26B" && return 0; done; return 1; }
BCBC="BENCHMARK=BCB $B CONFIG_ONLY=CLIMB-Live-BCB THETA_LIVE=99"

# ---------- C: wait for the Gemma runs started by part 6 (MuSiQue shards 0-1 only, no repeats), merge ----------
sleep 30
while gemma_alive; do sleep 60; done
G=traces/tasks_sweep_gemma-4-26B-A4B-think
for c in climbx gx4k; do
    cat ${G}_${c}s{0,1}_fr100.jsonl > ${G}_${c}_fr100.jsonl
    cat ${G}_${c}s{0,1}_mu100.jsonl > ${G}_${c}_mu100.jsonl
    cat ${G}_${c}s{0,1,2,3}_bcb100.jsonl > ${G}_${c}_bcb100.jsonl
done
wc -l ${G}_{climbx,gx4k}_{ma,fr,mu,bcb}100.jsonl
log "C done (Gemma)"

# ---------- D: gpt-oss-120b judge + its BCB climb rerun ----------
redo_bcb() {   # $1 model; archive the prose-era climb files, rerun 4 shards on the given ports
    local m=$1; shift; local T=traces/tasks_sweep_$m
    mkdir -p _archive/traces_bcb_prose
    mv ${T}_climbx_bcb100.jsonl ${T}_climbxs?_bcb100.jsonl logs/${m}_climbxs?_bcb100.log _archive/traces_bcb_prose/ 2>/dev/null
    local i=0; for p in "$@"; do MODEL=$m PORT=$p MAX_WORKERS=96 bash chains/_gptoss_run.sh _climbxs${i}_bcb100 $BCBC TASK_IDS_FILE=$I/bcb_shard$i.json & i=$((i+1)); done
}
tunnels 8001 $P8001
switch serve_gpt_oss_120b.sh 18101 gpt-oss-120b || exit 1
redo_bcb gpt-oss-120b 18111 18112 18113 18114
FILES=""
for t in climbx_fr100 climbx_mu100 gx4k_fr100 gx4k_mu100 gx4kr2_fr100 gx4kr2_mu100; do FILES="$FILES,gpt-oss-20b_$t"; done
for t in climbx_fr100 climbx_mu100 gx4k_fr100 gx4k_mu100; do FILES="$FILES,gemma-4-26B-A4B-think_$t"; done
FILES="${FILES#,},gpt-oss-120b_gx4kr2_fr100,gpt-oss-120b_gx4kr2_mu100"
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_overnight.jsonl JUDGE_THREADS=192 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JUDGE_FILES=$FILES
(.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply; log "D judge done") &
wait
T=traces/tasks_sweep_gpt-oss-120b; cat ${T}_climbxs{0,1,2,3}_bcb100.jsonl > ${T}_climbx_bcb100.jsonl; wc -l ${T}_climbx_bcb100.jsonl
log "D done (120b)"

# ---------- E: gpt-oss-20b BCB climb rerun ----------
switch serve_gpt_oss_20b.sh 18101 gpt-oss-20b || exit 1
redo_bcb gpt-oss-20b 18111 18112 18113 18114
wait
T=traces/tasks_sweep_gpt-oss-20b; cat ${T}_climbxs{0,1,2,3}_bcb100.jsonl > ${T}_climbx_bcb100.jsonl; wc -l ${T}_climbx_bcb100.jsonl
log "E done (20b)"

sed "s#glob.glob('traces/tasks_sweep_\*.jsonl')#glob.glob('traces/tasks_sweep_*_ma100.jsonl')#" analysis/rescore_math.py > $S/rescore_ma100.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python $S/rescore_ma100.py | tail -1
SUFFIX=100 .venv/bin/python analysis/export_records_new.py gpt-oss-120b:gpt-oss-120b gpt-oss-20b:gpt-oss-20b gemma-4-26B-A4B-think:Gemma-4-26B-A4B
log "OVERNIGHT DONE"
