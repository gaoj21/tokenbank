#!/bin/bash
# DeepSeek, structured reviewer at cap 4096 (equal to all other calls), MATH and GAIA 24k.
cd /home/j0018/Desktop/tokenbank
while ! grep -q "ABL DONE" logs/_chain_ablslot_ds.log 2>/dev/null; do sleep 60; done
T=DeepSeek-V4.1-Flash
for b in "MATH MATH_N=500:ma" "GAIA GAIA_N=1000:ga"; do
  set -- ${b%%:*}; t=${b##*:}; tag=_dsrv4k_${t}24
  [ -s traces/tasks_sweep_$T$tag.jsonl ] && continue
  echo "[$(date '+%F %T')] START $tag"
  env OPENAI_BASE_URL=http://127.0.0.1:18000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=deepseek-v4.1-flash MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 \
    MAX_WORKERS=128 MAX_TOKENS=4096 OUT_TAG=$tag BENCHMARK=$1 $2 BUDGETS=24000 SPLIT=test CONFIG_ONLY=ST-CritOnly-R4k,ST-Slots-R4k \
    .venv/bin/python -u _budget_sweep_v2.py > logs/$tag.log 2>&1
  echo "[$(date '+%F %T')] DONE  $tag"
done
echo "[$(date '+%F %T')] RV4K DONE"
