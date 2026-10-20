#!/bin/bash
# DeepSeek-V4.1-Flash over the SSH tunnel, configured exactly as GLM was.
#
# Moved off OpenRouter, where the same model costs $1.19 per thousand item-arms and the
# credit left would have covered MATH and BigCodeBench only. This deployment is free, so the
# family gets the full five-benchmark table. The $1.28 already spent there bought the
# per-item price and a working probe; those cells are under _aborted/ds_openrouter/.
#
# One difference from the OpenRouter path is worth recording rather than ignoring: there,
# logprobs covered `content` alone, while here they cover the reasoning trace as well (25
# tokens against a 19-token trace plus a short answer on the same probe). phi is therefore
# a different quantity on the two paths, and the one used here is the same one GLM's numbers
# were computed from, which is the reason to prefer this path for comparability rather than
# only for cost.
#
# Arms, budgets and ordering follow GLM: three arms on the main table, the BCB convention on
# BigCodeBench with its own budgets, ReasonLadder carrying l01_reasoning_effort=low. Whether
# reasoning_effort does nothing on this server, which the first completed cell settled: the
# ladder arm scored 22% non-empty answers against 20% for the arm carrying no rung at all,
# where GLM at the same cell goes 71% to 98%. The cheap rung is therefore the chat-template
# switch (L01_THINKING_OFF), one of only two knobs the server honours. The other is a hard
# token budget, which cuts mid-thought and so pays in truncation; the switch starts no
# thought at all and writes the answer directly, which is the cleaner rung. The superseded
# cells and the knob probe are under _aborted/ds_effort_nominal/.
set -e
cd /home/j0018/Desktop/tokenbank
TAG=DeepSeek-V4.1-Flash
URL=http://127.0.0.1:18000/v1
THREE="V5-Unconstrained,V5-Unc-Fmt,V5.2-ReasonLadder"
BCBARMS="BCB-Naive,BCB-Ab7,BCB-ReasonLadder"
base() {
  env OPENAI_BASE_URL=$URL VLLM_API_KEY=EMPTY VLLM_MODEL=deepseek-v4.1-flash MODEL_TAG=$TAG \
      SEEDS=42 MAX_WORKERS=256 TEMPERATURE=0 V5=1 L01_THINKING_OFF=1 "$@"
}
run() {
  local bench=$1 budget=$2 tag=$3 extra=$4
  [ -s "tasks_sweep_$TAG$tag.jsonl" ] && { echo "SKIP $tag"; return; }
  echo "[$(date '+%F %T')] START $tag"
  base BENCHMARK=$bench BUDGETS=$budget SPLIT=test CONFIG_ONLY="$THREE" \
       OUT_TAG=$tag $extra .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
run_bcb() {
  local budget=$1 tag=$2
  [ -s "tasks_sweep_$TAG$tag.jsonl" ] && { echo "SKIP $tag"; return; }
  echo "[$(date '+%F %T')] START $tag"
  base BENCHMARK=BCB BCB_N=500 BUDGETS=$budget SPLIT=full V5_MAX_ROUNDS=5 \
       CONFIG_ONLY="$BCBARMS" OUT_TAG=$tag DUMP_RAW_ANSWER=1 \
       .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
}
for B in 4000 8000 16000; do run_bcb $B _ds_bcb${B%000}k; done
for B in 24000 16000 8000; do run MATH    $B _ds_ma${B%000} MATH_N=500; done
for B in 24000 16000 8000; do run GAIA    $B _ds_ga${B%000} GAIA_N=1000; done
for B in 24000 16000 8000; do run FRAMES  $B _ds_fr${B%000} x=1; done
for B in 24000 16000 8000; do run MUSIQUE $B _ds_mu${B%000} MUSIQUE_N=99999; done
echo "[$(date '+%F %T')] DEEPSEEK ALL DONE"
