#!/bin/bash
# Re-evaluate all three arms with the reply-to-region alignment in place.
# The constructor previously substituted a fragment reply for the whole region,
# deleting the lines around it; 20% of the governed arms' patched files then
# failed to parse, against 8.5% for the baseline. The fix aligns a reply to the
# span it covers and keeps the rest, is applied identically to every arm, and
# leaves gold unchanged on 491 of 500 instances with 9/9 of the rest still
# resolving.
set -e
cd /home/j0018/Desktop/tokenbank
for spec in "naive swe_preds_naive_spliced.json" "ours swe_preds_ours_spliced.json" "ab7 swe_preds_ab7fix_spliced.json"; do
  set -- $spec; n=$1; p=$2
  echo "[$(date '+%F %T')] START $n"
  .venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified --predictions_path "$p" \
    --max_workers 14 --run_id "sp_$n" > "logs/_swe_sp_$n.log" 2>&1 || echo "  (rc=$?)"
  echo "[$(date '+%F %T')] DONE  $n"
  grep -E "Instances (completed|resolved)" "logs/_swe_sp_$n.log" | sed 's/^/    /'
done
echo "[$(date '+%F %T')] SWE SPLICE DONE"
