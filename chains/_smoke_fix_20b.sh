#!/bin/bash
# Smoke test of the tool-loop fix on gpt-oss-20b: 60 tasks per benchmark, forced climb and reference.
cd /home/j0018/Desktop/tokenbank
I=results/climb_live; B="BUDGETS=100000 SPLIT=test"; export MODEL=gpt-oss-20b
r() { PORT=$1 MAX_WORKERS=60 bash chains/_gptoss_run.sh "${@:2}" & }
r 18111 _fxsm_climb_ma BENCHMARK=MATH MATH_N=500 $B CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 TASK_IDS_FILE=$I/smoke60_ma.json
r 18112 _fxsm_ref_ma BENCHMARK=MATH MATH_N=500 $B CONFIG_ONLY=V5-Unc-Fmt TASK_IDS_FILE=$I/smoke60_ma.json
r 18113 _fxsm_climb_fr BENCHMARK=FRAMES $B CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 TASK_IDS_FILE=$I/smoke60_fr.json
r 18114 _fxsm_ref_fr BENCHMARK=FRAMES $B CONFIG_ONLY=V5-Unc-Fmt TASK_IDS_FILE=$I/smoke60_fr.json
r 18115 _fxsm_climb_mu BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 TASK_IDS_FILE=$I/smoke60_mu.json
r 18116 _fxsm_ref_mu BENCHMARK=MUSIQUE MUSIQUE_N=99999 $B CONFIG_ONLY=V5-Unc-Fmt TASK_IDS_FILE=$I/smoke60_mu.json
wait
