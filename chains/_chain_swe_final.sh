#!/bin/bash
# Final SWE evaluation: all three arms re-scored with the corrected parser.
#
# Two problems invalidated the first pass. The Ab7 arm inherited an
# answer-format instruction the other arms did not have, which conflicts with
# the fenced-code output this benchmark needs; and the reply parser required
# "### REGION" headers, discarding a bare code block whenever the instance had
# more than one hunk. The single-call arm produced a code block on all 500
# instances and labelled only 118, so most of its work was thrown away by the
# parser rather than missing. All three arms are re-parsed and re-evaluated
# together, since the old numbers were produced under the stricter parser.
set -e
cd /home/j0018/Desktop/tokenbank
for spec in "naive swe_preds_naive.json" "ours swe_preds_ours.json" "ab7 swe_preds_ab7fix.json"; do
  set -- $spec; name=$1; preds=$2
  echo "[$(date '+%F %T')] START $name"
  .venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified --predictions_path "$preds" \
    --max_workers 14 --run_id "fin_$name" > "logs/_swe_fin_$name.log" 2>&1 || echo "  (rc=$?)"
  echo "[$(date '+%F %T')] DONE  $name"
  grep -E "Instances (completed|resolved)" "logs/_swe_fin_$name.log" | sed 's/^/    /'
done
echo "[$(date '+%F %T')] SWE FINAL DONE"
