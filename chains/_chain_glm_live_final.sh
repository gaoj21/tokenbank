#!/bin/bash
# GLM FRAMES / MuSiQue: recalibrate on the FP8 records and run live on FP8. Stops on any failure.
set -e
cd /home/j0018/Desktop/tokenbank
G=GLM-5.3-Flash; I=results/climb_live
M="BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test"; F="BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test"
# single-call shards (appendix) are still running on llm3-5; wait so each endpoint runs one job at a time
until [ -s traces/tasks_sweep_${G}_glmfp8s_mu24x0.jsonl ] && [ -s traces/tasks_sweep_${G}_glmfp8s_mu24x1.jsonl ] && [ -s traces/tasks_sweep_${G}_glmfp8s_mu24x2.jsonl ]; do sleep 20; done
cat traces/tasks_sweep_${G}_glmfp8s_mu24x{0,1,2}.jsonl > traces/tasks_sweep_${G}_glmfp8s_mu24.jsonl
echo "[$(date '+%F %T')] single MuSiQue tasks=$(wc -l < traces/tasks_sweep_${G}_glmfp8s_mu24.jsonl) APIERR=$(cat logs/${G}_glmfp8s_mu24x*.log | grep -c APIERR || true)"
.venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, 'analysis')
from risk_gate import CELLS, items
from climb_calib import calibrate
P = json.load(open('results/climb_live_plan.json'))
out = []
for ds, ae, nf in CELLS['GLM']:
    if ds not in ('FRAMES', 'MuSiQue'): continue
    k = f'GLM|{ds}'; off = {x['id']: x for x in items(ae, nf, keep_id=True)}
    cal = [off[i] for i in P[k]['cal_ids'] if i in off]
    assert len(cal) == len(P[k]['cal_ids']), (k, len(cal))
    P[k]['threshold_v3'] = calibrate(cal, 0.05); out.append(str(P[k]['threshold_v3'])); print(k, len(cal), P[k]['threshold_v3'])
json.dump(P, open('results/climb_live_plan.json', 'w'))
open('results/climb_live/glm_fix_thresholds.txt', 'w').write(' '.join(out))
PY
read TF TM < results/climb_live/glm_fix_thresholds.txt
[ -n "$TF" ] && [ -n "$TM" ] || { echo "empty threshold, abort"; exit 1; }
echo "[$(date '+%F %T')] thresholds FRAMES=$TF MuSiQue=$TM"
run() { local t=$1 h=$2 th=$3 ids=$4; shift 4
  echo "[$(date '+%F %T')] START $t ($h) theta=$th"
  env OPENAI_BASE_URL=https://$h.ruoyu-wang.com/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=$G SEEDS=42 TEMPERATURE=0 V5=1 \
    MAX_WORKERS=32 MAX_TOKENS=4096 OUT_TAG=$t CONFIG_ONLY=CLIMB-Live THETA_LIVE=$th TASK_IDS_FILE=$ids "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$G$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t APIERR=$(grep -c APIERR logs/$G$t.log || true)"; }
.venv/bin/python -c "
import json; ids=json.load(open('results/climb_live/test_ids_GLM_MuSiQue.json'))
[json.dump(ids[k::2],open(f'results/climb_live/live_mu_x{k}.json','w')) for k in range(2)]"
set +e
if [ "$TF" != "direct" ]; then run _climb3_fr24 llm3 $TF $I/test_ids_GLM_FRAMES.json $F & fi
if [ "$TM" != "direct" ]; then run _climb3_mu24x0 llm4 $TM $I/live_mu_x0.json $M & run _climb3_mu24x1 llm5 $TM $I/live_mu_x1.json $M & fi
wait
[ -s traces/tasks_sweep_${G}_climb3_mu24x0.jsonl ] && cat traces/tasks_sweep_${G}_climb3_mu24x{0,1}.jsonl > traces/tasks_sweep_${G}_climb3_mu24.jsonl
echo "[$(date '+%F %T')] GLM LIVE FINAL DONE"
