#!/bin/bash
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] START goldcheck7"
.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite --predictions_path swe_preds_gold.json \
  --max_workers 4 --run_id goldcheck7 \
  --instance_ids astropy__astropy-12907 astropy__astropy-14182 django__django-11039 \
                 sympy__sympy-20049 scikit-learn__scikit-learn-13497 pytest-dev__pytest-5227 \
  > logs/_swe_goldcheck7.log 2>&1
echo "[$(date '+%F %T')] DONE goldcheck7"
