#!/bin/bash
# gpt-oss-120b: forced-climb records and the reference pipeline on MATH, FRAMES (2 shards) and MuSiQue
# (4 shards). Climb runs request log-probs and use compressed tunnels; shards are merged at the end.
cd /home/j0018/Desktop/tokenbank
R=chains/_gptoss_run.sh; export MAX_WORKERS=96 MODEL=gpt-oss-120b I=results/climb_live
run() { PORT=$1 bash $R "${@:2}" & }
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99"; P="CONFIG_ONLY=V5-Unc-Fmt"; B="BUDGETS=100000 SPLIT=test"
run 18111 _climbx_ma100 BENCHMARK=MATH MATH_N=500 $B $C
run 18112 _climbxs0_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half0.json
run 18113 _climbxs1_fr100 BENCHMARK=FRAMES $B $C TASK_IDS_FILE=$I/fr_half1.json
i=0; for port in 18114 18115 18116 18117; do run $port _climbxs${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $C TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
run 18118 _gx4k_ma100 BENCHMARK=MATH MATH_N=500 $B $P
run 18119 _gx4ks0_fr100 BENCHMARK=FRAMES $B $P TASK_IDS_FILE=$I/fr_half0.json
run 18120 _gx4ks1_fr100 BENCHMARK=FRAMES $B $P TASK_IDS_FILE=$I/fr_half1.json
i=0; for port in 18102 18104 18106 18108; do run $port _gx4ks${i}_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B $P TASK_IDS_FILE=$I/oss_mu_shard$i.json; i=$((i+1)); done
wait
T=traces/tasks_sweep_gpt-oss-120b
cat ${T}_climbxs{0,1}_fr100.jsonl > ${T}_climbx_fr100.jsonl; cat ${T}_gx4ks{0,1}_fr100.jsonl > ${T}_gx4k_fr100.jsonl
cat ${T}_climbxs{0,1,2,3}_mu100.jsonl > ${T}_climbx_mu100.jsonl; cat ${T}_gx4ks{0,1,2,3}_mu100.jsonl > ${T}_gx4k_mu100.jsonl
wc -l ${T}_{climbx,gx4k}_{ma,fr,mu}100.jsonl
echo "[$(date '+%F %T')] GPTOSS120B DONE"
