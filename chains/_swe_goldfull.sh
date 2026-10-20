#!/bin/bash
# Gold patches for all 500 Verified instances. Two purposes at once:
#   1. complete validation of the diff constructor. Anything below 500/500
#      resolved is an edge case the 14 sampled instances did not reach.
#   2. image warming. Every instance image is pulled here, which is the slow
#      part of evaluation; the two real arms afterwards then run from cache.
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] START goldfull"
.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified \
  --predictions_path swe_preds_verified_gold.json \
  --max_workers 8 --run_id goldfull \
  > logs/_swe_goldfull.log 2>&1
echo "[$(date '+%F %T')] DONE goldfull rc=$?"
