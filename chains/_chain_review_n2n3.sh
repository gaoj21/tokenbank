#!/bin/bash
# Review N2 (full, GLM): live CLIMB on the test halves with thresholds certified on the calibration
# halves under the corrected labels (results/climb_live/n2_plan.json), paired with a fresh run of the
# reference pipeline, three independent rounds. GAIA and FRAMES certify 'direct' (CLIMB = reference).
# Review N3 (GLM): single call, default reasoning, 16k generation cap, same budget, all four test splits.
cd /home/j0018/Desktop/tokenbank
G=GLM-5.3-Flash; M=GLM-5.3-Flash-FP8; I=results/climb_live
run() { local t=$1 url=$2; shift 2
  [ -s traces/tasks_sweep_$G$t.jsonl ] && { echo SKIP $t; return; }
  echo "[$(date '+%F %T')] START $t"
  env OPENAI_BASE_URL=$url VLLM_API_KEY=EMPTY VLLM_MODEL=$M MODEL_TAG=$G SEEDS=42 TEMPERATURE=0 V5=1 MAX_WORKERS=32 \
    MAX_TOKENS=4096 OUT_TAG=$t "$@" .venv/bin/python -u _budget_sweep_v2.py > logs/$G$t.log 2>&1
  echo "[$(date '+%F %T')] DONE  $t"; }
ma() { run $1 $2 BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test TASK_IDS_FILE=$I/n2_test_ids_GLM_MATH.json "${@:3}"; }
mu() { run $1 $2 BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test TASK_IDS_FILE=$I/n2_test_ids_GLM_MuSiQue.json "${@:3}"; }
L3=https://llm3.ruoyu-wang.com/v1; L5=https://llm5.ruoyu-wang.com/v1; L2=https://llm2.ruoyu-wang.com/v1
( for r in 1 2 3; do ma _n2c${r}_ma24 $L3 CONFIG_ONLY=CLIMB-Live THETA_LIVE=-0.05; ma _n2p${r}_ma24 $L3 CONFIG_ONLY=V5-Unc-Fmt; done
  mu _n2c3_mu24 $L3 CONFIG_ONLY=CLIMB-Live THETA_LIVE=-0.05; mu _n2p3_mu24 $L3 CONFIG_ONLY=V5-Unc-Fmt ) &
( for r in 1 2; do mu _n2c${r}_mu24 $L5 CONFIG_ONLY=CLIMB-Live THETA_LIVE=-0.05; mu _n2p${r}_mu24 $L5 CONFIG_ONLY=V5-Unc-Fmt; done ) &
( for b in MATH:ma:MATH_N=500 GAIA:ga:GAIA_N=1000 FRAMES:fr:X=0 MUSIQUE:mu:MUSIQUE_N=99999; do IFS=: read B a n <<< "$b"
    run _n3s16k_${a}24 $L2 BENCHMARK=$B $n BUDGETS=24000 SPLIT=test CONFIG_ONLY=V5-Single-Fmt-16k MAX_TOKENS=16384; done ) &
wait; echo "[$(date '+%F %T')] REVIEW N2/N3 DONE"
