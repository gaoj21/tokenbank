#!/bin/bash
# Gemma-4-31B-it cross-model run, mirroring the Qwen main table.
#
# Scope note: a full mirror is 3,116 items x 3 arms x 3 budgets = 28,044 runs.
# At the measured aggregate throughput of this server (159-205 tok/s at 16
# concurrent slots) and the Qwen table's 7,750 mean tokens per run, that is
# ~217M tokens and about 14 days. This chain therefore runs all four benchmarks
# at the headline 24k budget only, which is what the cross-model claim in the
# abstract needs, cheapest benchmark first so the table fills progressively.
# Extending to 8k/16k is a separate chain.
#
# The server must be started with --reasoning off. Newer llama.cpp builds enable
# thinking by default and put it in message.reasoning_content, which this
# harness never reads; the model would spend its whole per-call cap thinking and
# return an empty answer. That also matches the Qwen runs, which disable
# thinking via chat_template_kwargs.
set -e
cd /home/j0018/Desktop/tokenbank
ARMS="V5-Unconstrained,V5-Unc-Fmt,V5.1-Ab7-NoVerbal"
run() {
  local bench=$1 budget=$2 tag=$3 extra=$4
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=http://127.0.0.1:8001/v1 VLLM_MODEL=gemma-4-31B-it \
    MODEL_TAG=gemma-4-31B-it \
    BENCHMARK=$bench BUDGETS=$budget SEEDS=42 MAX_WORKERS=16 SPLIT=test \
    V5=1 CONFIG_ONLY="$ARMS" OUT_TAG=$tag TEMPERATURE=0 $extra \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
run GAIA    24000 _gem38_ga24 GAIA_N=1000
run MATH    24000 _gem38_ma24 MATH_N=500
run FRAMES  24000 _gem38_fr24 x=1
run MUSIQUE 24000 _gem38_mu24 MUSIQUE_N=99999
echo "[$(date '+%F %T')] GEMMA MAIN DONE"
