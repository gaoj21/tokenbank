#!/bin/bash
# gpt-oss-120b phase 2 (no per-task budget): BigCodeBench forced-climb and reference (4 shards each),
# and a second, independent run of the reference pipeline on 300 test tasks per benchmark, to separate
# systematic team harm from sampling variation.
cd /home/j0018/Desktop/tokenbank
R=chains/_gptoss_run.sh; export MODEL=gpt-oss-120b; I=results/climb_live; B="BUDGETS=100000 SPLIT=test"
run() { PORT=$1 MAX_WORKERS=$2 bash $R "${@:3}" & }
i=0; for port in 18111 18112 18113 18114; do run $port 64 _climbxs${i}_bcb100 BENCHMARK=BCB $B CONFIG_ONLY=CLIMB-Live-BCB THETA_LIVE=99 TASK_IDS_FILE=$I/bcb_shard$i.json; i=$((i+1)); done
i=0; for port in 18115 18116 18117 18118; do run $port 64 _gx4ks${i}_bcb100 BENCHMARK=BCB $B CONFIG_ONLY=BCB-Naive TASK_IDS_FILE=$I/bcb_shard$i.json; i=$((i+1)); done
run 18119 200 _gx4kr2_ma100 BENCHMARK=MATH MATH_N=500 $B CONFIG_ONLY=V5-Unc-Fmt TASK_IDS_FILE=$I/repeat300_ma.json
run 18120 200 _gx4kr2_fr100 BENCHMARK=FRAMES $B CONFIG_ONLY=V5-Unc-Fmt TASK_IDS_FILE=$I/repeat300_fr.json
run 18102 200 _gx4kr2_mu100 BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B CONFIG_ONLY=V5-Unc-Fmt TASK_IDS_FILE=$I/repeat300_mu.json
run 18104 64 _gx4kr2_bcb100 BENCHMARK=BCB $B CONFIG_ONLY=BCB-Naive TASK_IDS_FILE=$I/repeat300_bcb.json
wait
T=traces/tasks_sweep_gpt-oss-120b
cat ${T}_climbxs{0,1,2,3}_bcb100.jsonl > ${T}_climbx_bcb100.jsonl; cat ${T}_gx4ks{0,1,2,3}_bcb100.jsonl > ${T}_gx4k_bcb100.jsonl
wc -l ${T}_{climbx,gx4k}_bcb100.jsonl ${T}_gx4kr2_*100.jsonl
echo "[$(date '+%F %T')] GPTOSS120B PHASE2 DONE"
