#!/bin/bash
cd /home/j0018/Desktop/tokenbank
for f in GLM DeepSeek; do for d in MATH GAIA FRAMES MuSiQue; do .venv/bin/python analysis/eps_sweep.py $f $d > results/eps_sweep_${f}_${d}.log 2>&1 & done; done
wait; echo "[$(date '+%F %T')] EPS SWEEP DONE"
