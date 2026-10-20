#!/bin/bash
# GLM-5.3-Flash analysis experiments, on the same SGLang deployment as the main table (five
# endpoints, verified one deployment). One shard per endpoint, run in parallel.
# Pipeline arms (Naive+Fmt, all-escalate) run with MAX_TOKENS=4096 from the start: on
# DeepSeek the 1024 per-call cap cost the baseline 4.2 points on MATH and 13.9 on GAIA, so
# repeating the 1024 version would only re-measure the artefact. GLM thinking cannot be
# disabled, so the ladder's cheap rung is reasoning_effort=low, as in the main table.
cd /home/j0018/Desktop/tokenbank
TAG=GLM-5.3-Flash
job() { local url=$1 model=$2; shift 2
  for spec in "$@"; do
    IFS='|' read t env arms <<< "$spec"
    [ -s "traces/tasks_sweep_$TAG$t.jsonl" ] && { echo "SKIP $t"; continue; }
    echo "[$(date '+%F %T')] START $t @ $url"
    env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$TAG SEEDS=42 \
        MAX_WORKERS=32 TEMPERATURE=0 V5=1 OUT_TAG=$t CONFIG_ONLY=$arms $env \
        .venv/bin/python -u _budget_sweep_v2.py > logs/$t.log 2>&1
    echo "[$(date '+%F %T')] DONE  $t"
  done; }
M="BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test"
G="BENCHMARK=GAIA GAIA_N=1000 BUDGETS=24000 SPLIT=test"
F="BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test"
U="BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test"
C="BENCHMARK=BCB BCB_N=500 BUDGETS=16000 SPLIT=full V5_MAX_ROUNDS=5 DUMP_RAW_ANSWER=1"
S=V5-Single-Fmt-1k,V5-Single-Fmt-2k; P=V5-Unc-Fmt,V5.2-ReasonLadder-AllEsc
job https://llm3.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 \
  "_glmx_single_ma24|$M|$S" "_glmx4k_ma24|$M MAX_TOKENS=4096|$P" \
  "_glmx_single_ga24|$G|$S" "_glmx4k_ga24|$G MAX_TOKENS=4096|$P" &
job https://llm5.ruoyu-wang.com/v1 GLM-5.3-Flash-FP8 \
  "_glmx_single_fr24|$F|$S" "_glmx4k_fr24|$F MAX_TOKENS=4096|$P" &
job https://llm52.jierungogogo.com/v1 zai-org/GLM-5.3-Flash \
  "_glmx_single_mu24|$U|$S" "_glmx4kn_mu24|$U MAX_TOKENS=4096|V5-Unc-Fmt" &
job https://llm56.jierungogogo.com/v1 zai-org/GLM-5.3-Flash \
  "_glmx4ka_mu24|$U MAX_TOKENS=4096|V5.2-ReasonLadder-AllEsc" &
job https://llm57.jierungogogo.com/v1 zai-org/GLM-5.3-Flash \
  "_glmx_bcbsingle16k|$C|BCB-Single-1k,BCB-Single-2k" \
  "_glmx4k_bcb16k|$C MAX_TOKENS=4096|BCB-Naive,BCB-ReasonLadder-AllEsc" &
wait
echo "[$(date '+%F %T')] GLM ANALYSIS ALL DONE"
