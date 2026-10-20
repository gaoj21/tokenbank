#!/bin/bash
# Once the reference MuSiQue and the MATH/GAIA climbs finish, restart the gpt-oss-20b MuSiQue climb
# as four shards (one compressed tunnel each) to use the freed server capacity, then merge.
cd /home/j0018/Desktop/tokenbank
until grep -q "DONE  _gx4k_mu24" logs/_chain_gptoss20b.log && [ -s traces/tasks_sweep_gpt-oss-20b_climbx_ma24.jsonl ] && [ -s traces/tasks_sweep_gpt-oss-20b_climbx_ga24.jsonl ]; do sleep 20; done
for p in $(pgrep -f _budget_sweep_v2.py); do e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null); echo "$e" | grep -q "^MODEL_TAG=gpt-oss-20b$" && echo "$e" | grep -q "^OUT_TAG=_climbx_mu24$" && kill $p && echo "[$(date '+%F %T')] stopped single MuSiQue climb $p"; done
sleep 5; f=traces/tasks_sweep_gpt-oss-20b_climbx_mu24.jsonl; [ -f $f ] && [ ! -s $f ] && rm $f
R=chains/_gptoss_run.sh; export MAX_WORKERS=128 MODEL=gpt-oss-20b
i=0; for port in 18111 18112 18115 18116; do
  PORT=$port bash $R _climbxs${i}_mu24 BENCHMARK=MUSIQUE MUSIQUE_N=99999 BUDGETS=24000 SPLIT=test CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 TASK_IDS_FILE=results/climb_live/oss_mu_shard$i.json &
  i=$((i+1)); done
wait
cat traces/tasks_sweep_gpt-oss-20b_climbxs{0,1,2,3}_mu24.jsonl > traces/tasks_sweep_gpt-oss-20b_climbx_mu24.jsonl
echo "[$(date '+%F %T')] MU SHARDS MERGED $(wc -l < traces/tasks_sweep_gpt-oss-20b_climbx_mu24.jsonl)"
