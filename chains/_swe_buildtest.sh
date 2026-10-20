#!/bin/bash
# swebench 3.0.17 builds images locally instead of pulling prebuilt ones, so it
# is not subject to Docker Hub's 100-pull-per-6h anonymous limit that turned
# 402 of 500 instances into "errors" under 5.0.2. --namespace none forces the
# local build path; --cache_level env keeps the per-repo environment images
# (shared by many instances) and discards the per-instance layers after use.
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] START buildtest"
/tmp/claude-1001/-home-j0018-Desktop-tokenbank/f14fef7e-adf9-443b-9307-5c07475168b4/scratchpad/swebench3/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path swe_preds_verified_gold.json \
  --max_workers 2 --run_id buildtest --namespace none --cache_level env \
  --instance_ids astropy__astropy-12907 django__django-10097 \
  > logs/_swe_buildtest.log 2>&1
echo "[$(date '+%F %T')] DONE buildtest rc=$?"
