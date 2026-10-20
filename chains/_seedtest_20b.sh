#!/bin/bash
# Draft-seeded escalation with a verify instruction (SEED_PROMPT=verify), gpt-oss-20b, forced climb,
# same settings as the fixed 20b reruns; compared with the fresh reference (gx4k) and the old seeding.
cd /home/j0018/Desktop/tokenbank
I=results/climb_live; B="BUDGETS=100000 SPLIT=test L01_MAX_TOKENS=8192 SEED_PROMPT=verify"; export MODEL=gpt-oss-20b
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"; BC="CONFIG_ONLY=CLIMB-Live-BCB THETA_LIVE=99"
r() { PORT=$1 MAX_WORKERS=$2 bash chains/_gptoss_run.sh "${@:3}" & }
r 18111 128 _seedv_ma100 BENCHMARK=MATH MATH_N=500 $B $C
r 18112 128 _seedvs0_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half0.json
r 18113 128 _seedvs1_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half1.json
r 18114 128 _seedv_mu0100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $C TASK_IDS_FILE=$I/oss_mu_shard0.json
i=0; for p in 18115 18116 18117 18118; do r $p 96 _seedvs${i}_bcb100 BENCHMARK=BCB $B $BC TASK_IDS_FILE=$I/bcb_shard$i.json; i=$((i+1)); done
wait
T=traces/tasks_sweep_gpt-oss-20b
cat ${T}_seedvs{0,1}_fr100.jsonl > ${T}_seedv_fr100.jsonl; cat ${T}_seedvs{0,1,2,3}_bcb100.jsonl > ${T}_seedv_bcb100.jsonl
wc -l ${T}_seedv_{ma,fr,bcb}100.jsonl ${T}_seedv_mu0100.jsonl
echo "[$(date '+%F %T')] SEEDTEST DONE"
