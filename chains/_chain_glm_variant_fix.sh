#!/bin/bash
# Variant fix for GLM: every run of a benchmark uses the model variant of its reference pipeline
# (FRAMES: FP8 on llm2-5; MuSiQue: BF16 on llm52-57). Re-record, recalibrate, rerun live.
cd /home/j0018/Desktop/tokenbank
run() { local T=$1 t=$2 url=$3 model=$4 w=$5 cfg=$6 extra=$7; shift 7
  echo "[$(date '+%F %T')] START $T$t ($url)"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$model MODEL_TAG=$T SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=$w \
    MAX_TOKENS=4096 OUT_TAG=$t CONFIG_ONLY=$cfg $extra "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/${T}$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $T$t"; }
G=GLM-5.3-Flash; I=results/climb_live
FP8=GLM-5.3-Flash-FP8; BF=zai-org/GLM-5.3-Flash
F="BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test"; M="BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test"
# 1) records (THETA=99) + 2) contemporaneous pipelines, in parallel
k=0; for h in llm2 llm3 llm4 llm5; do run $G _climbx_fr24x$k https://$h.ruoyu-wang.com/v1 $FP8 32 CLIMB-Live "THETA_LIVE=99 TASK_IDS_FILE=$I/fr_x$k.json" $F & k=$((k+1)); done
k=0; for h in llm53 llm54 llm55 llm57; do run $G _climbx_mu24x$k https://$h.jierungogogo.com/v1 $BF 32 CLIMB-Live "THETA_LIVE=99 TASK_IDS_FILE=$I/mu_x$k.json" $M & k=$((k+1)); done
run $G _pipe3_fr24 https://llm2.ruoyu-wang.com/v1 $FP8 24 V5-Unc-Fmt "TASK_IDS_FILE=$I/test_ids_GLM_FRAMES.json" $F &
run $G _pipe3_mu24s1 https://llm52.jierungogogo.com/v1 $BF 32 V5-Unc-Fmt "TASK_IDS_FILE=$I/pipe_mu_shard1.json" $M &
run $G _pipe3_mu24s2 https://llm57.jierungogogo.com/v1 $BF 24 V5-Unc-Fmt "TASK_IDS_FILE=$I/pipe_mu_shard2.json" $M &
wait
cat traces/tasks_sweep_${G}_climbx_fr24x{0,1,2,3}.jsonl > traces/tasks_sweep_${G}_climbx_fr24.jsonl
cat traces/tasks_sweep_${G}_climbx_mu24s0.jsonl traces/tasks_sweep_${G}_climbx_mu24x{0,1,2,3}.jsonl > traces/tasks_sweep_${G}_climbx_mu24.jsonl
until [ -s traces/tasks_sweep_${G}_pipe3_mu24s0.jsonl ]; do sleep 30; done
cat traces/tasks_sweep_${G}_pipe3_mu24s{0,1,2}.jsonl > traces/tasks_sweep_${G}_pipe3_mu24.jsonl
wc -l traces/tasks_sweep_${G}_climbx_fr24.jsonl traces/tasks_sweep_${G}_climbx_mu24.jsonl traces/tasks_sweep_${G}_pipe3_*24.jsonl
echo "[$(date '+%F %T')] RECORDS FIXED"
# 3) recalibrate GLM FRAMES / MuSiQue on the same calibration halves
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
    P[k]['threshold_v3'] = calibrate(cal, 0.05); print(k, len(cal), P[k]['threshold_v3'])
json.dump(P, open('results/climb_live_plan.json', 'w'))
open('results/climb_live/glm_fix_thresholds.txt', 'w').write(' '.join(str(P[f'GLM|{d}']['threshold_v3']) for d in ('FRAMES', 'MuSiQue')))
PY
read TF TM < results/climb_live/glm_fix_thresholds.txt
echo "thresholds FRAMES=$TF MuSiQue=$TM"
.venv/bin/python -c "
import json; ids=json.load(open('results/climb_live/test_ids_GLM_MuSiQue.json'))
[json.dump(ids[k::3],open(f'results/climb_live/live_mu_x{k}.json','w')) for k in range(3)]"
[ "$TF" != "direct" ] && run $G _climb3_fr24 https://llm4.ruoyu-wang.com/v1 $FP8 32 CLIMB-Live "THETA_LIVE=$TF TASK_IDS_FILE=$I/test_ids_GLM_FRAMES.json" $F &
if [ "$TM" != "direct" ]; then k=0; for h in llm53 llm54 llm55; do run $G _climb3_mu24x$k https://$h.jierungogogo.com/v1 $BF 32 CLIMB-Live "THETA_LIVE=$TM TASK_IDS_FILE=$I/live_mu_x$k.json" $M & k=$((k+1)); done; fi
wait
[ -s traces/tasks_sweep_${G}_climb3_mu24x0.jsonl ] && cat traces/tasks_sweep_${G}_climb3_mu24x{0,1,2}.jsonl > traces/tasks_sweep_${G}_climb3_mu24.jsonl
echo "[$(date '+%F %T')] GLM VARIANT FIX DONE"
