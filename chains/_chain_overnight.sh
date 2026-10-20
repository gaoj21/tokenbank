#!/bin/bash
# Overnight relay (no per-task budget, 100k safety cap):
#   A. wait for gpt-oss-120b phase 2 and its FRAMES/MuSiQue judge;
#   B. gpt-oss-20b: forced climb + reference on MATH/FRAMES/MuSiQue/BigCodeBench, reference repeat on 300/bench;
#   C. Gemma-4-31B (thinking): BCB smoke, then the same runs (BCB only if the smoke passes), repeats last;
#   D. gpt-oss-120b again as judge for 20b, Gemma and the 120b repeats; MATH rescoring; records export.
# Each step checks line counts instead of trusting DONE.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
I=results/climb_live; B="BUDGETS=100000 SPLIT=test"
log() { echo "[$(date '+%F %T')] $*"; }

tunnels() {   # re-open any dropped tunnel: $1 remote port, rest local ports
    local rp=$1; shift
    for p in "$@"; do
        ss -ltn | grep -q ":$p " && continue
        setsid nohup ssh -C -i $K -p 820 -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
            -N -L $p:127.0.0.1:$rp root@47.82.65.107 >/dev/null 2>&1 < /dev/null &
        sleep 4
    done
}
P8001="$(seq 18101 18108) $(seq 18111 18130)"; P8000="$(seq 18000 18008) $(seq 18010 18029)"

switch() {    # $1 serve script, $2 local port to poll, $3 served model name
    log "switch server -> $1"
    timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done
        for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done
        cd /mnt/data/models && (setsid nohup bash '$1' > '${1%.sh}'.log 2>&1 < /dev/null &); echo started'
    for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:$2/v1/models | grep -q "$3" && { log "server ready: $3"; return 0; }; sleep 15; done
    log "SERVER FAILED: $3"; return 1
}

# ---------- A ----------
until grep -q "PHASE2 DONE" logs/_chain_gptoss120b_phase2.log && grep -q "JUDGE 120B DONE" logs/_judge_oss120b.log; do sleep 60; done
log "A done"

# ---------- B: gpt-oss-20b ----------
tunnels 8001 $P8001
switch serve_gpt_oss_20b.sh 18101 gpt-oss-20b || exit 1
export MODEL=gpt-oss-20b; R=chains/_gptoss_run.sh
run() { PORT=$1 MAX_WORKERS=$2 bash $R "${@:3}" & }
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"; F="CONFIG_ONLY=V5-Unc-Fmt"
run 18111 128 _climbx_ma100 BENCHMARK=MATH MATH_N=500 $B $C
run 18112 128 _climbxs0_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half0.json
run 18113 128 _climbxs1_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half1.json
i=0; for p in 18114 18115 18116 18117; do run $p 128 _climbxs${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $C TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
i=0; for p in 18118 18119 18120 18121; do run $p 64 _climbxs${i}_bcb100 BENCHMARK=BCB $B CONFIG_ONLY=CLIMB-Live-BCB THETA_LIVE=99 TASK_IDS_FILE=$I/bcb_shard$i.json; i=$((i+1)); done
run 18122 128 _gx4k_ma100 BENCHMARK=MATH MATH_N=500 $B $F
run 18123 128 _gx4ks0_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/fr_half0.json
run 18124 128 _gx4ks1_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/fr_half1.json
i=0; for p in 18125 18126 18127 18128; do run $p 128 _gx4ks${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
i=0; for p in 18129 18130 18101 18102; do run $p 64 _gx4ks${i}_bcb100 BENCHMARK=BCB $B CONFIG_ONLY=BCB-Naive TASK_IDS_FILE=$I/bcb_shard$i.json; i=$((i+1)); done
run 18103 128 _gx4kr2_ma100 BENCHMARK=MATH MATH_N=500 $B $F TASK_IDS_FILE=$I/repeat300_ma.json
run 18104 128 _gx4kr2_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/repeat300_fr.json
run 18105 128 _gx4kr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/repeat300_mu.json
run 18106 64 _gx4kr2_bcb100 BENCHMARK=BCB $B CONFIG_ONLY=BCB-Naive TASK_IDS_FILE=$I/repeat300_bcb.json
wait
T=traces/tasks_sweep_gpt-oss-20b
for c in climbx gx4k; do
    cat ${T}_${c}s{0,1}_fr100.jsonl > ${T}_${c}_fr100.jsonl
    cat ${T}_${c}s{0,1,2,3}_mu100.jsonl > ${T}_${c}_mu100.jsonl
    cat ${T}_${c}s{0,1,2,3}_bcb100.jsonl > ${T}_${c}_bcb100.jsonl
done
wc -l ${T}_{climbx,gx4k}_{ma,fr,mu,bcb}100.jsonl ${T}_gx4kr2_*100.jsonl
log "B done (gpt-oss-20b)"

# ---------- C: Gemma-4-31B ----------
tunnels 8000 $P8000
switch serve_gemma4_31b_reason_tp2.sh 18000 gemma-4-31b-it || exit 1
R=chains/_gemma_run.sh
grun() { PORT=$1 MAX_WORKERS=$2 bash $R "${@:3}" TOP_LOGPROBS=0 & }
grun 18010 24 _bcbsmokeclimb BENCHMARK=BCB $B CONFIG_ONLY=CLIMB-Live-BCB THETA_LIVE=99 TASK_IDS_FILE=$I/bcb_smoke24.json
grun 18000 24 _bcbsmokeref BENCHMARK=BCB $B CONFIG_ONLY=BCB-Naive TASK_IDS_FILE=$I/bcb_smoke24.json
wait
G=traces/tasks_sweep_gemma-4-31B-think
BCB_OK=$(.venv/bin/python - <<PY
import json
def rd(p):
    try: return [json.loads(l) for l in open(p)]
    except Exception: return []
c, r = rd('${G}_bcbsmokeclimb.jsonl'), rd('${G}_bcbsmokeref.jsonl')
ok = len(c) == 24 and len(r) == 24 and sum(x['success'] for x in r) >= 3 and sum(bool((x.get('raw_answer') or '').strip()) for x in r) >= 20
print(1 if ok else 0)
PY
)
log "Gemma BCB smoke ok=$BCB_OK ($(grep -c '"success": true' ${G}_bcbsmokeref.jsonl)/24 ref correct)"
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"
grun 18011 48 _climbx_ma100 BENCHMARK=MATH MATH_N=500 $B $C
grun 18012 48 _climbxs0_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half0.json
grun 18013 48 _climbxs1_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half1.json
i=0; for p in 18014 18015 18016 18017; do grun $p 48 _climbxs${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $C TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
grun 18021 48 _gx4k_ma100 BENCHMARK=MATH MATH_N=500 $B $F
grun 18022 48 _gx4ks0_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/fr_half0.json
grun 18023 48 _gx4ks1_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/fr_half1.json
i=0; for p in 18024 18025 18026 18027; do grun $p 48 _gx4ks${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
if [ "$BCB_OK" = 1 ]; then
    i=0; for p in 18018 18019 18020 18028; do grun $p 32 _climbxs${i}_bcb100 BENCHMARK=BCB $B CONFIG_ONLY=CLIMB-Live-BCB THETA_LIVE=99 TASK_IDS_FILE=$I/bcb_shard$i.json; i=$((i+1)); done
    i=0; for p in 18029 18001 18002 18003; do grun $p 32 _gx4ks${i}_bcb100 BENCHMARK=BCB $B CONFIG_ONLY=BCB-Naive TASK_IDS_FILE=$I/bcb_shard$i.json; i=$((i+1)); done
fi
wait
log "C main done (Gemma)"
grun 18004 64 _gx4kr2_ma100 BENCHMARK=MATH MATH_N=500 $B $F TASK_IDS_FILE=$I/repeat300_ma.json
grun 18005 64 _gx4kr2_fr100 BENCHMARK=FRAMES $B $F TASK_IDS_FILE=$I/repeat300_fr.json
grun 18006 64 _gx4kr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $F TASK_IDS_FILE=$I/repeat300_mu.json
[ "$BCB_OK" = 1 ] && grun 18007 48 _gx4kr2_bcb100 BENCHMARK=BCB $B CONFIG_ONLY=BCB-Naive TASK_IDS_FILE=$I/repeat300_bcb.json
wait
for c in climbx gx4k; do
    cat ${G}_${c}s{0,1}_fr100.jsonl > ${G}_${c}_fr100.jsonl
    cat ${G}_${c}s{0,1,2,3}_mu100.jsonl > ${G}_${c}_mu100.jsonl
    [ "$BCB_OK" = 1 ] && cat ${G}_${c}s{0,1,2,3}_bcb100.jsonl > ${G}_${c}_bcb100.jsonl
done
wc -l ${G}_{climbx,gx4k}_*100.jsonl ${G}_gx4kr2_*100.jsonl
log "C done (Gemma)"

# ---------- D: judge with gpt-oss-120b, rescore MATH, export ----------
tunnels 8001 $P8001
switch serve_gpt_oss_120b.sh 18101 gpt-oss-120b || exit 1
FILES=""
for m in gpt-oss-20b gemma-4-31B-think; do for t in climbx_fr100 climbx_mu100 gx4k_fr100 gx4k_mu100 gx4kr2_fr100 gx4kr2_mu100; do FILES="$FILES,${m}_$t"; done; done
FILES="${FILES#,},gpt-oss-120b_gx4kr2_fr100,gpt-oss-120b_gx4kr2_mu100"
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_overnight.jsonl JUDGE_THREADS=192 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JUDGE_FILES=$FILES
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply
log "D judge done"
sed "s#glob.glob('traces/tasks_sweep_\*.jsonl')#glob.glob('traces/tasks_sweep_*_ma100.jsonl')#" analysis/rescore_math.py > /tmp/rescore_ma100.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python /tmp/rescore_ma100.py | tail -1
SUFFIX=100 .venv/bin/python analysis/export_records_new.py gpt-oss-120b:gpt-oss-120b gpt-oss-20b:gpt-oss-20b gemma-4-31B-think:Gemma-4-31B
log "OVERNIGHT DONE"
