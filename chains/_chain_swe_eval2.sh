#!/bin/bash
# The remaining three arms at higher concurrency. The first chain used 8 workers
# while the BigCodeBench re-judging also held the machine; that finished, so the
# 20 cores are free. Gold is left to complete under the original chain because
# restarting it would discard 435 finished instances.
set -e
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] waiting for the gold arm"
while ! grep -q "DONE  eval_gold" logs/_swe_eval_chain.log 2>/dev/null; do
  pgrep -f 'chain_swe_eval[.]sh' >/dev/null || break
  sleep 60
done
# stop the original chain before it starts the arms at 8 workers
pkill -9 -f 'chain_swe_eval[.]sh' 2>/dev/null || true
sleep 5
for spec in "naive swe_preds_naive.json" "ours swe_preds_ours.json" "ab7 swe_preds_ab7.json"; do
  set -- $spec; name=$1; preds=$2
  echo "[$(date '+%F %T')] START eval_$name"
  .venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified --predictions_path "$preds" \
    --max_workers 14 --run_id "ev_$name" > "logs/_swe_eval_$name.log" 2>&1 || echo "  (rc=$?)"
  echo "[$(date '+%F %T')] DONE  eval_$name"
  grep -E "Instances (resolved|unresolved|completed|with errors)" "logs/_swe_eval_$name.log" | sed 's/^/    /'
done
echo "[$(date '+%F %T')] SWE EVAL ARMS DONE"
