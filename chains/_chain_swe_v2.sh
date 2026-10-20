#!/bin/bash
# SWE, regenerated after the fourth defect. The bounded-plan slot kept
# plan_lines[1] verbatim, which on repair plans is usually a heading
# ("**Step-by-Step Plan:**") rather than a step, so the slot carried nothing on
# exactly the task type where the plan matters. Both the pipeline and the slot
# ablation now skip scaffolding lines and keep the first line that is a step.
# All three arms are regenerated together; the previous run is not comparable.
set -e
cd /home/j0018/Desktop/tokenbank
source ./_llm5_env.sh
echo "[$(date '+%F %T')] START sweo_v2 (3 arms)"
env BENCHMARK=SWEO SWE_N=500 BUDGETS=24000 SEEDS=42 MAX_WORKERS=32 SPLIT=full \
  V5=1 V5_MAX_ROUNDS=20 CONFIG_ONLY="SWEO-Naive,SWEO-Ours,SWEO-Ab7" \
  OUT_TAG=_sweo_v2 TEMPERATURE=0 DUMP_RAW_ANSWER=1 \
  .venv/bin/python -u _budget_sweep_v2.py > logs/_sweo_v2.log 2>&1
echo "[$(date '+%F %T')] DONE sweo_v2  (API errors: $(grep -c '\[APIERR\]' logs/_sweo_v2.log))"
echo "[$(date '+%F %T')] START slot_v2"
N=250 W=24 .venv/bin/python -u _swe_slot_ablation.py > logs/_swe_slot_v2.log 2>&1
echo "[$(date '+%F %T')] DONE slot_v2"
grep -E "usable|mean plan|non-empty" logs/_swe_slot_v2.log
