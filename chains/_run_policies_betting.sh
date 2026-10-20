#!/bin/bash
# Policy comparison re-run with the final calibration (betting), 10 x 20 reps.
cd /home/j0018/Desktop/tokenbank
for i in 0 1 2 3 4 5 6 7 8 9; do PART=policies REPS=20 SEED=$i OUT=results/abl_parts_pol_betting/pol_$i.json .venv/bin/python analysis/ablations_glm.py > results/abl_parts_pol_betting/pol_$i.log 2>&1 & done
wait; echo "[$(date '+%F %T')] POLICIES BETTING DONE"
