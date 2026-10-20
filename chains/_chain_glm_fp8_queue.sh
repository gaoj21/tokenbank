#!/bin/bash
# GLM, FP8 only (llm2-5), one 32-worker job per endpoint at a time (a full request queue on the
# server would turn calls into errors). Jobs per endpoint run sequentially; FRAMES shards x1-x3 are
# already running on llm3-5 and are waited for. Then recalibrate FRAMES/MuSiQue and run them live.
cd /home/j0018/Desktop/tokenbank
G=GLM-5.3-Flash; I=results/climb_live
M="BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test"; F="BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test"
run() { local t=$1 h=$2 cfg=$3 extra=$4; shift 4
  echo "[$(date '+%F %T')] START $t ($h)"
  env OPENAI_BASE_URL=https://$h.ruoyu-wang.com/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=GLM-5.3-Flash-FP8 MODEL_TAG=$G SEEDS=42 TEMPERATURE=0 V5=1 \
    MAX_WORKERS=32 MAX_TOKENS=4096 OUT_TAG=$t CONFIG_ONLY=$cfg $extra "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$G$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t ($h) APIERR=$(grep -c APIERR logs/$G$t.log)"; }
waitfile() { until [ -s traces/tasks_sweep_${G}$1.jsonl ]; do sleep 30; done; }
( run _climbx_fr24x0 llm2 CLIMB-Live "THETA_LIVE=99 TASK_IDS_FILE=$I/fr_x0.json" $F
  run _glmfp8n_mu24x0 llm2 V5-Unc-Fmt "TASK_IDS_FILE=$I/mu8_0.json" $M
  run _climbx_mu24f0 llm2 CLIMB-Live "THETA_LIVE=99 TASK_IDS_FILE=$I/mu8_0.json" $M
  run _glmfp8s_mu24 llm2 V5-Single-Fmt-4k "" $M ) &
( waitfile _climbx_fr24x1
  run _glmfp8n_mu24x1 llm3 V5-Unc-Fmt "TASK_IDS_FILE=$I/mu8_1.json" $M
  run _climbx_mu24f1 llm3 CLIMB-Live "THETA_LIVE=99 TASK_IDS_FILE=$I/mu8_1.json" $M
  run _pipe3_fr24 llm3 V5-Unc-Fmt "TASK_IDS_FILE=$I/test_ids_GLM_FRAMES.json" $F ) &
( waitfile _climbx_fr24x2
  run _glmfp8n_mu24x2 llm4 V5-Unc-Fmt "TASK_IDS_FILE=$I/mu8_2.json" $M
  run _climbx_mu24f2 llm4 CLIMB-Live "THETA_LIVE=99 TASK_IDS_FILE=$I/mu8_2.json" $M ) &
( waitfile _climbx_fr24x3
  run _glmfp8n_mu24x3 llm5 V5-Unc-Fmt "TASK_IDS_FILE=$I/mu8_3.json" $M
  run _climbx_mu24f3 llm5 CLIMB-Live "THETA_LIVE=99 TASK_IDS_FILE=$I/mu8_3.json" $M ) &
wait
cat traces/tasks_sweep_${G}_climbx_fr24x{0,1,2,3}.jsonl > traces/tasks_sweep_${G}_climbx_fr24.jsonl
cat traces/tasks_sweep_${G}_climbx_mu24f{0,1,2,3}.jsonl > traces/tasks_sweep_${G}_climbx_mu24.jsonl
cat traces/tasks_sweep_${G}_glmfp8n_mu24x{0,1,2,3}.jsonl > traces/tasks_sweep_${G}_glmfp8n_mu24.jsonl
wc -l traces/tasks_sweep_${G}_climbx_fr24.jsonl traces/tasks_sweep_${G}_climbx_mu24.jsonl traces/tasks_sweep_${G}_glmfp8n_mu24.jsonl traces/tasks_sweep_${G}_glmfp8s_mu24.jsonl traces/tasks_sweep_${G}_pipe3_fr24.jsonl
echo "APIERR in FP8 FRAMES shards x1-x3: $(cat logs/${G}_climbx_fr24x{1,2,3}.log | grep -c APIERR)"
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
[ "$TF" != "direct" ] && run _climb3_fr24 llm5 CLIMB-Live "THETA_LIVE=$TF TASK_IDS_FILE=$I/test_ids_GLM_FRAMES.json" $F &
if [ "$TM" != "direct" ]; then k=0; for h in llm2 llm3 llm4; do run _climb3_mu24x$k $h CLIMB-Live "THETA_LIVE=$TM TASK_IDS_FILE=$I/live_mu_x$k.json" $M & k=$((k+1)); done; fi
wait
[ -s traces/tasks_sweep_${G}_climb3_mu24x0.jsonl ] && cat traces/tasks_sweep_${G}_climb3_mu24x{0,1,2}.jsonl > traces/tasks_sweep_${G}_climb3_mu24.jsonl
echo "[$(date '+%F %T')] GLM FP8 ALL DONE"
