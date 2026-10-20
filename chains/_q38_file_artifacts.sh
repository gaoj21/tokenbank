#!/bin/bash
# Qwen3.8 归档看守：产物写在根目录，跑完一个搬一个。
cd /home/j0018/Desktop/tokenbank
LOG=logs/_q38_file_artifacts.log
echo "[$(date '+%F %T')] 启动" >> $LOG
while pgrep -f "_chain_q38" >/dev/null 2>&1; do
  for f in budget_sweep_v2_Qwen3.8-27B_*.json; do
    [ -e "$f" ] || continue
    t="tasks_sweep_Qwen3.8-27B_${f#budget_sweep_v2_Qwen3.8-27B_}"; t="${t%.json}.jsonl"
    if [ -s "$t" ] && ! fuser "$t" >/dev/null 2>&1; then
      n=$(wc -l < "$t")
      mv "$f" results/ && mv "$t" traces/ && echo "[$(date '+%F %T')] 归档 $f (trace $n 行)" >> $LOG
    fi
  done
  sleep 120
done
sleep 20
for f in budget_sweep_v2_Qwen3.8-27B_*.json; do
  [ -e "$f" ] || continue
  t="tasks_sweep_Qwen3.8-27B_${f#budget_sweep_v2_Qwen3.8-27B_}"; t="${t%.json}.jsonl"
  mv "$f" results/ 2>/dev/null; [ -s "$t" ] && mv "$t" traces/ 2>/dev/null
done
echo "[$(date '+%F %T')] 结束" >> $LOG
