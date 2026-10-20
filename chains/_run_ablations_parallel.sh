#!/bin/bash
# GLM ablations split over 15 processes (10 x 20 reps of the policy comparison, 5 x 40 reps of
# the calibration-design comparison), each with its own seed; merged by analysis/merge_ablations.py.
cd /home/j0018/Desktop/tokenbank
for i in 0 1 2 3 4 5 6 7 8 9; do PART=policies REPS=20 SEED=$i OUT=results/abl_parts/pol_$i.json .venv/bin/python analysis/ablations_glm.py > results/abl_parts/pol_$i.log 2>&1 & done
for i in 0 1 2 3 4; do PART=calib REPS=40 SEED=$i OUT=results/abl_parts/cal_$i.json .venv/bin/python analysis/ablations_glm.py > results/abl_parts/cal_$i.log 2>&1 & done
wait; echo "[$(date '+%F %T')] ABLATIONS DONE"
