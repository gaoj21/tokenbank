#!/bin/bash
# Gemma-4-31B-it with thinking: deployed-CLIMB records (forced climb) and the reference pipeline
# on the four test splits, 24k budget, 4,096 per-call cap; same protocol as GLM and DeepSeek except
# the pipeline's sampling (see _gemma_run.sh).
cd /home/j0018/Desktop/tokenbank
R=chains/_gemma_run.sh; export MAX_WORKERS=128
# One SSH tunnel per run (local ports 18001-18008): a single tunnel cannot forward ~1,000
# concurrent connections, and most requests never reached the server.
port=18001
for b in MATH:ma:MATH_N=500 GAIA:ga:GAIA_N=1000 FRAMES:fr:X=0 MUSIQUE:mu:MUSIQUE_N=99999; do
  IFS=: read B a n <<< "$b"
  PORT=$port bash $R _climbx_${a}24 BENCHMARK=$B $n BUDGETS=24000 SPLIT=test CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 &
  PORT=$((port+1)) bash $R _gx4k_${a}24 BENCHMARK=$B $n BUDGETS=24000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt &
  port=$((port+2))
done
wait; echo "[$(date '+%F %T')] GEMMA THINK DONE"
