#!/bin/bash
# Final offline recomputation on the FP8-consistent GLM records (betting calibration):
# main replay table (both models), GLM ablations (policies, calibration design), GLM appendix.
cd /home/j0018/Desktop/tokenbank
MAIN_OUT=results/main_table_final.json .venv/bin/python analysis/main_table.py > results/final_parts/main.log 2>&1 &
for i in 0 1 2 3 4 5 6 7 8 9; do PART=policies REPS=20 SEED=$i OUT=results/final_parts/pol_$i.json .venv/bin/python analysis/ablations_glm.py > results/final_parts/pol_$i.log 2>&1 & done
for i in 0 1 2 3 4; do PART=calib REPS=40 SEED=$i OUT=results/final_parts/cal_$i.json .venv/bin/python analysis/ablations_glm.py > results/final_parts/cal_$i.log 2>&1 & done
.venv/bin/python analysis/appendix_glm.py > results/final_parts/appendix.log 2>&1 &
wait
echo "[$(date '+%F %T')] FINAL CPU DONE"
