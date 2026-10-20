#!/bin/bash
# Rerun gpt-oss-20b and gpt-oss-120b on MATH/FRAMES/MuSiQue after the tool-loop fix (cheap rungs and
# pipeline executor may now run code up to three times, then one closing call without the tool).
# BigCodeBench is unaffected (the code is the answer) and is kept. Old traces go to
# _archive/traces_toolloop_bug/. Then judge FRAMES/MuSiQue with gpt-oss-120b, rescore MATH, export.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
I=results/climb_live; B="BUDGETS=100000 SPLIT=test L01_MAX_TOKENS=8192"; S=/tmp/claude-1001/-home-j0018-Desktop-tokenbank/f14fef7e-adf9-443b-9307-5c07475168b4/scratchpad
log() { echo "[$(date '+%F %T')] $*"; }
tunnels() { local rp=$1; shift; for p in "$@"; do ss -ltn | grep -q ":$p " && continue
    setsid nohup ssh -C -i $K -p 820 -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L $p:127.0.0.1:$rp root@47.82.65.107 >/dev/null 2>&1 < /dev/null & sleep 4; done; }
P8001="$(seq 18101 18108) $(seq 18111 18130)"
switch() {
    log "switch server -> $1"
    timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
        for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done
        cd /mnt/data/models && (setsid nohup bash '$1' > '${1%.sh}'.log 2>&1 < /dev/null &); echo started'
    for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:$2/v1/models | grep -q "$3" && { log "server ready: $3"; return 0; }; sleep 15; done
    log "SERVER FAILED: $3"; return 1
}
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"; F="CONFIG_ONLY=V5-Unc-Fmt"
runall() {   # $1 model
    local m=$1; local T=traces/tasks_sweep_$m
    mkdir -p _archive/traces_toolloop_bug
    for b in ma fr mu; do mv ${T}_climbx*_${b}100.jsonl ${T}_gx4k*_${b}100.jsonl logs/${m}_climbx*_${b}100.log logs/${m}_gx4k*_${b}100.log _archive/traces_toolloop_bug/ 2>/dev/null; done
    r() { MODEL=$m PORT=$1 MAX_WORKERS=$2 bash chains/_gptoss_run.sh "${@:3}" & }
    r 18111 128 _climbx_ma100 BENCHMARK=MATH MATH_N=500 $B $C
    r 18112 128 _climbxs0_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half0.json
    r 18113 128 _climbxs1_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half1.json
    local i=0; for p in 18114 18115 18116 18117; do r $p 128 _climbxs${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $C TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
    r 18118 128 _gx4k_ma100 BENCHMARK=MATH MATH_N=500 $B $F
    r 18119 128 _gx4ks0_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/fr_half0.json
    r 18120 128 _gx4ks1_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/fr_half1.json
    i=0; for p in 18121 18122 18123 18124; do r $p 128 _gx4ks${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
    r 18125 128 _gx4kr2_ma100 BENCHMARK=MATH MATH_N=500 $B $F TASK_IDS_FILE=$I/repeat300_ma.json
    r 18126 128 _gx4kr2_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/repeat300_fr.json
    r 18127 128 _gx4kr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/repeat300_mu.json
    wait
    for c in climbx gx4k; do cat ${T}_${c}s{0,1}_fr100.jsonl > ${T}_${c}_fr100.jsonl; cat ${T}_${c}s{0,1,2,3}_mu100.jsonl > ${T}_${c}_mu100.jsonl; done
    wc -l ${T}_{climbx,gx4k}_{ma,fr,mu}100.jsonl ${T}_gx4kr2_{ma,fr,mu}100.jsonl
    log "$m done"
}
tunnels 8001 $P8001
runall gpt-oss-20b
switch serve_gpt_oss_120b.sh 18101 gpt-oss-120b || exit 1
runall gpt-oss-120b
FILES=""; for m in gpt-oss-20b gpt-oss-120b; do for t in climbx_fr100 climbx_mu100 gx4k_fr100 gx4k_mu100 gx4kr2_fr100 gx4kr2_mu100; do FILES="$FILES,${m}_$t"; done; done
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_fix.jsonl JUDGE_THREADS=256 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JUDGE_FILES=${FILES#,}
rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply
log "judge done"
sed "s#glob.glob('traces/tasks_sweep_\*.jsonl')#glob.glob('traces/tasks_sweep_gpt-oss-*_ma100.jsonl')#" analysis/rescore_math.py > $S/rescore_fix.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python $S/rescore_fix.py | tail -1
SUFFIX=100 .venv/bin/python analysis/export_records_new.py gpt-oss-120b:gpt-oss-120b gpt-oss-20b:gpt-oss-20b gemma-4-26B-A4B-qat:Gemma-4-26B-A4B
log "FIX GPT DONE"
