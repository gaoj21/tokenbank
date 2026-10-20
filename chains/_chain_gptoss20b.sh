#!/bin/bash
# gpt-oss-20b: forced-climb records and the reference pipeline on the four test splits, 24k budget,
# one SSH tunnel per run (local ports 18101-18108 -> remote 8001). Settings in _gptoss_run.sh.
cd /home/j0018/Desktop/tokenbank
R=chains/_gptoss_run.sh; export MAX_WORKERS=128 MODEL=gpt-oss-20b
port=18101
for b in MATH:ma:MATH_N=500 GAIA:ga:GAIA_N=1000 FRAMES:fr:X=0 MUSIQUE:mu:MUSIQUE_N=99999; do
  IFS=: read B a n <<< "$b"
  PORT=$port bash $R _climbx_${a}24 BENCHMARK=$B $n BUDGETS=24000 SPLIT=test CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 &
  PORT=$((port+1)) bash $R _gx4k_${a}24 BENCHMARK=$B $n BUDGETS=24000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt &
  port=$((port+2))
done
wait; echo "[$(date '+%F %T')] GPTOSS20B DONE"
