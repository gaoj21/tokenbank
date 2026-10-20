#!/bin/bash
# E4: on-policy single-slot bounding. The decisive experiment for the paper's
# motivation.
#
# The counterfactual slot ablation swaps one slot at a fixed round of a
# trajectory the UNBOUNDED pipeline produced, so the planner wrote its plan to
# be read whole and the bounded arm sees an out-of-distribution context. That
# makes its numbers an upper bound on harm, not an estimate. Here each arm runs
# end to end under its own bound from the first call, so the planner writes
# under the cap it will be read under.
#
# Arms are V5-Unc-Fmt plus EXACTLY ONE bound. The control (V5-Unc-Fmt @16k)
# already exists for all four datasets and is not re-run.
# SEGMENT_ACCOUNTING is on, so the same run yields both halves of the claim:
# how many tokens the slot occupies, and what bounding it costs.
#
# Waits for the motivation and main-table chains so it does not contend with
# them for the one remote server.
set -e
cd /home/j0018/Desktop/tokenbank
echo "[$(date '+%F %T')] E4 waiting for running chains"
while pgrep -f "_chain_segacct|_chain_slotabl4|_chain_q38_full|_chain_q38_attrib" >/dev/null; do sleep 120; done
echo "[$(date '+%F %T')] E4 starting"
for spec in "GAIA GAIA_N=1000" "MATH MATH_N=500" "FRAMES x=1" "MUSIQUE MUSIQUE_N=99999"; do
  set -- $spec; b=$1; extra=$2
  echo "[$(date '+%F %T')] START e4_$b"
  env OPENAI_BASE_URL=https://llm9.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=Qwen/Qwen3.8-27B MODEL_TAG=Qwen3.8-27B \
    BENCHMARK=$b BUDGETS=16000 SEEDS=42 MAX_WORKERS=40 SPLIT=test \
    V5=1 CONFIG_ONLY="E4-BoundPlan,E4-BoundCritique,E4-BoundNotes" \
    OUT_TAG=_e4_$b TEMPERATURE=0 SEGMENT_ACCOUNTING=1 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/_e4_$b.log 2>&1
  echo "[$(date '+%F %T')] DONE  e4_$b"
done
echo "[$(date '+%F %T')] E4 ALL DONE"
