#!/bin/bash
# Every paper analysis on data/records_final.csv (MATH/BBH closed-book, FRAMES/MuSiQue open-book).
cd /home/j0018/Desktop/tokenbank/paper
log() { echo "[$(date '+%F %T')] $*"; }
P=../.venv/bin/python
$P analysis/main_results.py data/records_final.csv 2>&1 | grep -v Warn && $P analysis/make_tab_main.py && log "main done" || { log "MAIN FAILED"; exit 1; }
$P analysis/final_analyses.py 2>&1 | grep -v Warn | tail -12; log "final_analyses done"
$P analysis/guarantee_iid_final.py 2>&1 | grep -v Warn | tail -5; log "iid done"
$P analysis/appendix_extras.py 2>&1 | grep -v Warn | tail -6; log "ALL ANALYSES DONE"
