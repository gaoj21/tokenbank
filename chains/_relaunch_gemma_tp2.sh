#!/bin/bash
cd /home/j0018/Desktop/tokenbank
until curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q gemma-4-31b-it; do sleep 20; done
echo "[$(date '+%F %T')] server ready (TP2 x DP4)"
rm -f traces/tasks_sweep_gemma-4-31B-think_{climbx,gx4k}_*24.jsonl
./chains/_chain_gemma_think.sh > logs/_chain_gemma_think.log 2>&1
