#!/bin/bash
# GLM on one model variant (FP8, llm2-5) everywhere. FRAMES records/pipeline are already running on
# FP8; here MuSiQue is re-run on FP8 (reference pipeline, all-escalate records, single call), then
# FRAMES and MuSiQue are recalibrated and run live on FP8.
cd /home/j0018/Desktop/tokenbank
run() { local t=$1 url=$2 w=$3 cfg=$4 extra=$5; shift 5
  echo "[$(date '+%F %T')] START $t ($url)"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=GLM-5.3-Flash SEEDS=42 TEMPERATURE=0 V5=1 \
    MAX_WORKERS=$w MAX_TOKENS=4096 OUT_TAG=$t CONFIG_ONLY=$cfg $extra "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/GLM-5.3-Flash$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
G=GLM-5.3-Flash; I=results/climb_live
M="BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test"; F="BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test"
k=0; for h in llm2 llm3 llm4 llm5; do
  run _glmfp8n_mu24x$k https://$h.ruoyu-wang.com/v1 24 V5-Unc-Fmt "TASK_IDS_FILE=$I/mu8_$k.json" $M &
  run _climbx_mu24f$k https://$h.ruoyu-wang.com/v1 24 CLIMB-Live "THETA_LIVE=99 TASK_IDS_FILE=$I/mu8_$k.json" $M &
  k=$((k+1)); done
run _glmfp8s_mu24 https://llm2.ruoyu-wang.com/v1 16 V5-Single-Fmt-4k "" $M &
wait
# FRAMES jobs were started by the previous chain; wait until their traces exist and no FRAMES job is left
until [ -s traces/tasks_sweep_${G}_climbx_fr24x0.jsonl ] && [ -s traces/tasks_sweep_${G}_climbx_fr24x1.jsonl ] && [ -s traces/tasks_sweep_${G}_climbx_fr24x2.jsonl ] \
   && [ -s traces/tasks_sweep_${G}_climbx_fr24x3.jsonl ] && [ -s traces/tasks_sweep_${G}_pipe3_fr24.jsonl ]; do sleep 30; done
cat traces/tasks_sweep_${G}_climbx_fr24x{0,1,2,3}.jsonl > traces/tasks_sweep_${G}_climbx_fr24.jsonl
cat traces/tasks_sweep_${G}_climbx_mu24f{0,1,2,3}.jsonl > traces/tasks_sweep_${G}_climbx_mu24.jsonl
cat traces/tasks_sweep_${G}_glmfp8n_mu24x{0,1,2,3}.jsonl > traces/tasks_sweep_${G}_glmfp8n_mu24.jsonl
wc -l traces/tasks_sweep_${G}_climbx_fr24.jsonl traces/tasks_sweep_${G}_climbx_mu24.jsonl traces/tasks_sweep_${G}_glmfp8n_mu24.jsonl traces/tasks_sweep_${G}_glmfp8s_mu24.jsonl
echo "[$(date '+%F %T')] FP8 RECORDS DONE"
.venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items
from climb_calib import calibrate
P = json.load(open('results/climb_live_plan.json'))
for ds, ae, nf in CELLS['GLM']:
    if ds not in ('FRAMES', 'MuSiQue'): continue
    k = f'GLM|{ds}'; off = {x['id']: x for x in items(ae, nf, keep_id=True)}
    cal = [off[i] for i in P[k]['cal_ids'] if i in off]
    P[k]['threshold_v3'] = calibrate(cal, 0.05); print(k, nf, len(cal), P[k]['threshold_v3'])
json.dump(P, open('results/climb_live_plan.json', 'w'))
open('results/climb_live/glm_fix_thresholds.txt', 'w').write(' '.join(str(P[f'GLM|{d}']['threshold_v3']) for d in ('FRAMES', 'MuSiQue')))
PY
read TF TM < results/climb_live/glm_fix_thresholds.txt; echo "thresholds FRAMES=$TF MuSiQue=$TM"
.venv/bin/python -c "
import json; ids=json.load(open('results/climb_live/test_ids_GLM_MuSiQue.json'))
[json.dump(ids[k::3],open(f'results/climb_live/live_mu_x{k}.json','w')) for k in range(3)]"
[ "$TF" != "direct" ] && run _climb3_fr24 https://llm5.ruoyu-wang.com/v1 32 CLIMB-Live "THETA_LIVE=$TF TASK_IDS_FILE=$I/test_ids_GLM_FRAMES.json" $F &
if [ "$TM" != "direct" ]; then k=0; for h in llm2 llm3 llm4; do run _climb3_mu24x$k https://$h.ruoyu-wang.com/v1 32 CLIMB-Live "THETA_LIVE=$TM TASK_IDS_FILE=$I/live_mu_x$k.json" $M & k=$((k+1)); done; fi
wait
[ -s traces/tasks_sweep_${G}_climb3_mu24x0.jsonl ] && cat traces/tasks_sweep_${G}_climb3_mu24x{0,1,2}.jsonl > traces/tasks_sweep_${G}_climb3_mu24.jsonl
echo "[$(date '+%F %T')] GLM FP8 ALL DONE"
