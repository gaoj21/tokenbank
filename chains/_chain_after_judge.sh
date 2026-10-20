#!/bin/bash
# After the LLM judge: apply judgments to FRAMES/MuSiQue traces, then recompute every analysis.
cd /home/j0018/Desktop/tokenbank
# judge finished (including retries on other endpoints)
echo "[$(date "+%F %T")] judge complete"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python analysis/llm_judge.py apply 2>&1 | grep -v "Warn\|cached\|Found"
echo "[$(date '+%F %T')] applied"
rm -f results/final_parts/*
MAIN_OUT=results/main_table_final.json .venv/bin/python analysis/main_table.py > results/final_parts/main.log 2>&1 &
for f in GLM DeepSeek; do for d in MATH GAIA FRAMES MuSiQue; do .venv/bin/python analysis/eps_sweep.py $f $d > results/eps_sweep_${f}_${d}.log 2>&1 & done; done
for i in 0 1 2 3 4; do PART=policies REPS=40 SEED=$i OUT=results/final_parts/pol_$i.json .venv/bin/python analysis/ablations_glm.py > results/final_parts/pol_$i.log 2>&1 & done
for i in 0 1 2 3 4; do PART=calib REPS=40 SEED=$i OUT=results/final_parts/cal_$i.json .venv/bin/python analysis/ablations_glm.py > results/final_parts/cal_$i.log 2>&1 & done
.venv/bin/python analysis/appendix_glm.py > results/final_parts/appendix.log 2>&1 &
.venv/bin/python analysis/monotone_gain.py > results/final_parts/monotone.log 2>&1 &
wait
.venv/bin/python analysis/merge_ablations.py > /dev/null
echo "[$(date '+%F %T')] ALL ANALYSES DONE"
