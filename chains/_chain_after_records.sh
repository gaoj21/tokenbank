#!/bin/bash
# When every all-escalate record is in, recompute the main replay table (with baselines).
cd /home/j0018/Desktop/tokenbank
until grep -q "CLIMBX MU2 DONE" logs/_chain_climbx_mu2.log 2>/dev/null && grep -q "CLIMBX DS DONE" logs/_chain_climbx_ds.log 2>/dev/null; do sleep 30; done
echo "[$(date '+%F %T')] records complete"; wc -l traces/tasks_sweep_*_climbx_*24.jsonl
.venv/bin/python analysis/main_table.py > results/main_table.log 2>&1
echo "[$(date '+%F %T')] MAIN TABLE DONE"
