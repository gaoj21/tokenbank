#!/bin/bash
# Re-run every analysis on records_final.csv after the MATH judge fix.
cd /home/j0018/Desktop/tokenbank/paper
P=../.venv/bin/python
$P analysis/main_results.py data/records_final.csv > ../logs/main_results.log 2>&1 || { echo MAIN FAILED; exit 1; }
echo "[$(date '+%T')] main_results done"
$P analysis/final_analyses.py > ../logs/final_analyses.log 2>&1 &
$P analysis/guarantee_iid_final.py > ../logs/guarantee_iid_final.log 2>&1 &
$P analysis/appendix_extras.py > ../logs/appendix_extras.log 2>&1 &
wait
echo "[$(date '+%T')] ALL ANALYSES DONE"
