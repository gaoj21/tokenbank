#!/bin/bash
# gpt-oss-20b reference FRAMES as four shards (compressed tunnels 18117-18120), then merge.
cd /home/j0018/Desktop/tokenbank
R=chains/_gptoss_run.sh; export MAX_WORKERS=128 MODEL=gpt-oss-20b
i=0; for port in 18117 18118 18119 18120; do
  PORT=$port bash $R _gx4ks${i}_fr24 BENCHMARK=FRAMES BUDGETS=24000 SPLIT=test CONFIG_ONLY=V5-Unc-Fmt TASK_IDS_FILE=results/climb_live/oss_fr_shard$i.json &
  i=$((i+1)); done
wait
cat traces/tasks_sweep_gpt-oss-20b_gx4ks{0,1,2,3}_fr24.jsonl > traces/tasks_sweep_gpt-oss-20b_gx4k_fr24.jsonl
echo "[$(date '+%F %T')] FR REF SHARDS MERGED $(wc -l < traces/tasks_sweep_gpt-oss-20b_gx4k_fr24.jsonl)"
