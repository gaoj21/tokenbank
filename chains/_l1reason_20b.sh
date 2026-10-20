#!/bin/bash
# 1) let the gpt-oss-120b reference runs of the stopped final relay finish, merge its shards;
# 2) gpt-oss-20b, forced climb to L1 with L1 = single agent at full reasoning, from scratch (L1_MODE=reason),
#    on MATH, FRAMES, MuSiQue shard 0 and BigCodeBench. The reference runs are reused.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa; SSH="ssh -i $K -p 820 -o BatchMode=yes -o ConnectTimeout=20 root@47.82.65.107"
log() { echo "[$(date '+%F %T')] $*"; }
while ps -eo args | awk '$1 ~ /python/ && $0 ~ /_budget_sweep_v2/' | grep -q .; do sleep 30; done
T=traces/tasks_sweep_gpt-oss-120b
for c in climbx gx4k; do cat ${T}_${c}s{0,1}_fr100.jsonl > ${T}_${c}_fr100.jsonl; cat ${T}_${c}s{0,1,2,3}_mu100.jsonl > ${T}_${c}_mu100.jsonl; done
wc -l ${T}_{climbx,gx4k}_{ma,fr,mu}100.jsonl ${T}_gx4kr2_{ma,fr,mu}100.jsonl
log "120b runs merged"
timeout 300 $SSH 'for p in $(ps -eo pid,args | awk "\$3 ~ /\/vllm\$/ && \$4==\"serve\" {print \$1}"); do kill $p; done; for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1000" | wc -l); [ "$n" = 0 ] && break; sleep 5; done; cd /mnt/data/models && (setsid nohup bash serve_gpt_oss_20b.sh > serve_gpt_oss_20b.log 2>&1 < /dev/null &); echo started'
for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-20b && break; sleep 15; done
log "20b ready"
I=results/climb_live; B="BUDGETS=100000 SPLIT=test L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason"; export MODEL=gpt-oss-20b
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"; BC="CONFIG_ONLY=CLIMB-Live-BCB THETA_LIVE=99"
r() { PORT=$1 MAX_WORKERS=$2 bash chains/_gptoss_run.sh "${@:3}" & }
r 18111 128 _l1r_ma100 BENCHMARK=MATH MATH_N=500 $B $C
r 18112 128 _l1rs0_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half0.json
r 18113 128 _l1rs1_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half1.json
r 18114 128 _l1r_mu0100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $C TASK_IDS_FILE=$I/oss_mu_shard0.json
i=0; for p in 18115 18116 18117 18118; do r $p 96 _l1rs${i}_bcb100 BENCHMARK=BCB $B $BC TASK_IDS_FILE=$I/bcb_shard$i.json; i=$((i+1)); done
wait
T=traces/tasks_sweep_gpt-oss-20b
cat ${T}_l1rs{0,1}_fr100.jsonl > ${T}_l1r_fr100.jsonl; cat ${T}_l1rs{0,1,2,3}_bcb100.jsonl > ${T}_l1r_bcb100.jsonl
wc -l ${T}_l1r_{ma,fr,bcb}100.jsonl ${T}_l1r_mu0100.jsonl
log "L1REASON DONE"
