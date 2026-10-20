#!/bin/bash
# Execution check for the MULTI-FILE code path, which the first gold check never
# exercised: all six of those instances patched a single file. 71/500 Verified
# instances touch more than one file, up to 21, and that assembly logic is new.
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] START goldmulti"
.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified --predictions_path swe_preds_verified_gold.json \
  --max_workers 6 --run_id goldmulti --instance_ids sympy__sympy-13091 sympy__sympy-16597 django__django-11532 django__django-14315 django__django-14376 django__django-14631 sympy__sympy-19783 sympy__sympy-22080 \
  > logs/_swe_goldmulti.log 2>&1
echo "[$(date '+%F %T')] DONE goldmulti"
