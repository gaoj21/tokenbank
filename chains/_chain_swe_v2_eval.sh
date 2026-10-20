#!/bin/bash
# Final SWE evaluation. All four implementation defects are fixed and the arms
# share one configuration: no answer-format instruction anywhere, the same reply
# parser, the same fragment-aligning constructor, and the same scaffolding-aware
# plan line selection in both the pipeline and the ablation.
set -e
cd /home/j0018/Desktop/tokenbank
for n in naive ours ab7; do
  echo "[$(date '+%F %T')] START $n"
  .venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified --predictions_path swe_preds_v2_$n.json \
    --max_workers 14 --run_id "v2_$n" > "logs/_swe_v2eval_$n.log" 2>&1 || echo "  (rc=$?)"
  echo "[$(date '+%F %T')] DONE  $n"
  grep -E "Instances (completed|resolved)" "logs/_swe_v2eval_$n.log" | sed 's/^/    /'
done
echo "[$(date '+%F %T')] SWE V2 EVAL DONE"
