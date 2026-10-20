#!/bin/bash
# The three arms, serially, at 14 workers. Fresh run ids: the previous attempt
# briefly ran two evaluations against the same run_id, because killing the old
# chain's shell left its python child alive, and both wrote into ev_naive.
set -e
cd /home/j0018/Desktop/tokenbank
for spec in "naive swe_preds_naive.json" "ours swe_preds_ours.json" "ab7 swe_preds_ab7.json"; do
  set -- $spec; name=$1; preds=$2
  echo "[$(date '+%F %T')] START eval_$name"
  .venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified --predictions_path "$preds" \
    --max_workers 14 --run_id "arm_$name" > "logs/_swe_eval_$name.log" 2>&1 || echo "  (rc=$?)"
  echo "[$(date '+%F %T')] DONE  eval_$name"
  grep -E "Instances (resolved|unresolved|completed|with errors)" "logs/_swe_eval_$name.log" | sed 's/^/    /'
done
echo "[$(date '+%F %T')] SWE ARMS DONE"
