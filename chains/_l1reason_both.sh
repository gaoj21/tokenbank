#!/bin/bash
# Use the spare gpt-oss-120b capacity now: run the L1_MODE=reason forced climb on 120b next to its last
# reference runs; when everything on 120b is done, merge, judge FRAMES/MuSiQue with 120b, then switch to
# 20b and run the same forced climb there.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
I=results/climb_live; B="BUDGETS=100000 SPLIT=test L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason"
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"; BC="CONFIG_ONLY=CLIMB-Live-BCB THETA_LIVE=99"
l1r() {   # $1 model, then 8 ports
    local m=$1; shift; local P=("$@")
    r() { MODEL=$m PORT=$1 MAX_WORKERS=$2 bash chains/_gptoss_run.sh "${@:3}" & }
    r ${P[0]} 128 _l1r_ma100 BENCHMARK=MATH MATH_N=500 $B $C
    r ${P[1]} 128 _l1rs0_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half0.json
    r ${P[2]} 128 _l1rs1_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half1.json
    r ${P[3]} 128 _l1r_mu0100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $C TASK_IDS_FILE=$I/oss_mu_shard0.json
    local i=0; for p in ${P[4]} ${P[5]} ${P[6]} ${P[7]}; do r $p 96 _l1rs${i}_bcb100 BENCHMARK=BCB $B $BC TASK_IDS_FILE=$I/bcb_shard$i.json; i=$((i+1)); done
}
merge_l1r() { local T=traces/tasks_sweep_$1; cat ${T}_l1rs{0,1}_fr100.jsonl > ${T}_l1r_fr100.jsonl; cat ${T}_l1rs{0,1,2,3}_bcb100.jsonl > ${T}_l1r_bcb100.jsonl; wc -l ${T}_l1r_{ma,fr,bcb}100.jsonl ${T}_l1r_mu0100.jsonl; }
sweeps() { ps -eo args | awk '$1 ~ /python/ && $0 ~ /_budget_sweep_v2/' | grep -q .; }

l1r gpt-oss-120b 18111 18112 18113 18114 18115 18116 18117 18118
log "120b L1-reason started"
while sweeps; do sleep 30; done
T=traces/tasks_sweep_gpt-oss-120b
for c in climbx gx4k; do cat ${T}_${c}s{0,1}_fr100.jsonl > ${T}_${c}_fr100.jsonl; cat ${T}_${c}s{0,1,2,3}_mu100.jsonl > ${T}_${c}_mu100.jsonl; done
wc -l ${T}_{climbx,gx4k}_{ma,fr,mu}100.jsonl ${T}_gx4kr2_{ma,fr,mu}100.jsonl
merge_l1r gpt-oss-120b
log "120b done"
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_l1r120.jsonl JUDGE_THREADS=256 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JUDGE_FILES=gpt-oss-120b_climbx_fr100,gpt-oss-120b_climbx_mu100,gpt-oss-120b_gx4k_fr100,gpt-oss-120b_gx4k_mu100,gpt-oss-120b_gx4kr2_fr100,gpt-oss-120b_gx4kr2_mu100,gpt-oss-120b_l1r_fr100,gpt-oss-120b_l1r_mu0100
rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || log "JUDGE 120B FAILED"
log "120b judged"
timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done; for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done; cd /mnt/data/models && (setsid nohup bash serve_gpt_oss_20b.sh > serve_gpt_oss_20b.log 2>&1 < /dev/null &); echo started'
for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-20b && break; sleep 15; done
log "20b ready"
l1r gpt-oss-20b 18111 18112 18113 18114 18115 18116 18117 18118
wait
merge_l1r gpt-oss-20b
log "L1REASON BOTH DONE"
