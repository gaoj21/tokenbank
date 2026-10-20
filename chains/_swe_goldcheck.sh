#!/bin/bash
# Validate the patch builder before trusting any real numbers: feed the harness
# the GOLD fix routed through our own diff constructor. If the builder is
# correct these must all resolve. Anything less means the constructor is
# dropping or mangling hunks, and every downstream number would be wrong.
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] START goldcheck"
.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path swe_preds_gold.json \
  --max_workers 4 \
  --run_id goldcheck6 \
  --instance_ids astropy__astropy-12907 astropy__astropy-14182 django__django-11039 \
                 sympy__sympy-20049 scikit-learn__scikit-learn-13497 pytest-dev__pytest-5227 \
  > logs/_swe_goldcheck.log 2>&1
echo "[$(date '+%F %T')] DONE goldcheck rc=$?"
