#!/bin/bash
# Is the carried plan read at all?
#
# The counterfactual ablation already shows that replacing the plan with its current step
# costs nothing, which establishes that the slot is compressible. It does not establish
# that its content is used. These variants hold the slot's length and shape roughly fixed
# and destroy its content instead: the same sentences shuffled, a plan written for a
# different question, and no plan at all. If accuracy is flat across all of them, the
# executor is not reading the plan, and what the bound removes is not redundancy but dead
# weight.
#
# The foreign plan comes from a fixed rotation over the sampled tasks, so every task lends
# and borrows exactly one plan.
set -e
cd /home/j0018/Desktop/tokenbank
run() {
  local bench=$1 n=$2 tag=$3
  [ -s "results/slot_ablation_$tag.json" ] && { echo "SKIP $tag"; return; }
  echo "[$(date '+%F %T')] START $tag ($bench, n=$n)"
  env OPENAI_BASE_URL=https://llm5.ruoyu-wang.com/v1 VLLM_API_KEY=sk-ruoyugogogo \
    VLLM_MODEL=qwen3.8-27b-fp8 BENCHMARK=$bench N=$n MAX_WORKERS=32 OUT_TAG=$tag \
    .venv/bin/python -u _slot_ablation.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
run MATH    250 cp_math
run GAIA    100 cp_gaia
run FRAMES  250 cp_frames
run MUSIQUE 250 cp_musique
echo "[$(date '+%F %T')] CORRUPT PLAN DONE"
