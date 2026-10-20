#!/bin/bash
# Locate the cause of the deficit on code repair.
#
# On SWE-bench Verified both governed arms resolve significantly fewer instances
# than the untreated pipeline (-9.0pp for the current system, p=1e-5), and the
# two governed arms do not differ, so the verbalized channel is not responsible
# and the bounded slots are the suspect. The 850 paired trials that found the
# plan slot free are all question answering; this runs the same counterfactual
# on repair, where a plan may carry the diagnosis rather than a decomposition.
set -e
cd /home/j0018/Desktop/tokenbank
source ./_llm5_env.sh
echo "[$(date '+%F %T')] START swe_slot_generation"
N=250 W=32 .venv/bin/python -u _swe_slot_ablation.py > logs/_swe_slot.log 2>&1
echo "[$(date '+%F %T')] DONE generation"
for v in control bounded_plan; do
  echo "[$(date '+%F %T')] START eval_slot_$v"
  .venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --predictions_path swe_preds_slot_$v.json \
    --max_workers 14 --run_id "slot_$v" > logs/_swe_eval_slot_$v.log 2>&1 || echo "  (rc=$?)"
  echo "[$(date '+%F %T')] DONE  eval_slot_$v"
  grep -E "Instances (resolved|unresolved)" logs/_swe_eval_slot_$v.log | sed 's/^/    /'
done
echo "[$(date '+%F %T')] SWE SLOT DONE"
