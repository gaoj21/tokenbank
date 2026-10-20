#!/bin/bash
cd /home/j0018/Desktop/tokenbank
I=results/climb_live; B="BUDGETS=10000000 SPLIT=test L01_MAX_TOKENS=8192 V5_MAX_LEVEL=1 L1_MODE=reason MAX_TOKENS=16384 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0"; export MODEL=gpt-oss-120b
PORT=18111 MAX_WORKERS=40 bash chains/_gptoss_run.sh _lcbsm_climb BENCHMARK=LCB $B CONFIG_ONLY=CLIMB-Live-BCB THETA_LIVE=99 TASK_IDS_FILE=$I/lcb_smoke40.json &
PORT=18112 MAX_WORKERS=40 bash chains/_gptoss_run.sh _lcbsm_ref BENCHMARK=LCB $B CONFIG_ONLY=BCB-Naive TASK_IDS_FILE=$I/lcb_smoke40.json &
wait
