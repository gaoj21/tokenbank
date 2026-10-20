#!/bin/bash
# Official SWE-bench Verified evaluation, all 500 instances, all local images.
#
# Gold runs first and is not optional: it is the external ground truth for the
# diff constructor. A gold patch that does not resolve means the constructor is
# wrong, and every arm's number would be wrong with it. The earlier partial run
# reached 97/98 with the single failure being a flaky PASS_TO_PASS test.
#
# Arms, in order: Naive (unbounded loop), Ours (the system as it was, with the
# verbalized escalation channel), Ab7 (the system as it now stands, routed by
# the log-probability gate alone).
set -e
cd /home/j0018/Desktop/tokenbank
for spec in "gold swe_preds_verified_gold.json" "naive swe_preds_naive.json" \
            "ours swe_preds_ours.json" "ab7 swe_preds_ab7.json"; do
  set -- $spec; name=$1; preds=$2
  echo "[$(date '+%F %T')] START eval_$name"
  .venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --predictions_path "$preds" \
    --max_workers 8 --run_id "ev_$name" \
    > "logs/_swe_eval_$name.log" 2>&1 || echo "  (rc=$?)"
  echo "[$(date '+%F %T')] DONE  eval_$name"
  grep -E "Instances (resolved|unresolved|completed|with errors)" "logs/_swe_eval_$name.log" | sed 's/^/    /'
done
echo "[$(date '+%F %T')] SWE EVAL ALL DONE"
