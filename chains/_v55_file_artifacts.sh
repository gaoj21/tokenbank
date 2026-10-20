#!/bin/bash
# v5.5 归档看守：sweep 把 trace/json 写在项目根目录，逐个跑完就移到 traces/ 和 results/。
# 脱离会话运行，链条整夜跑完也能自动归位。
cd /home/j0018/Desktop/tokenbank
LOG=logs/_v55_file_artifacts.log
echo "[$(date '+%F %T')] 看守启动" >> $LOG
while pgrep -f "_chain_v55_table" >/dev/null 2>&1; do
  for f in budget_sweep_v2_Qwen3.6-27B-FP8_v55_*.json; do
    [ -e "$f" ] || continue
    t="tasks_sweep_Qwen3.6-27B-FP8_${f#budget_sweep_v2_Qwen3.6-27B-FP8_}"; t="${t%.json}.jsonl"
    # 只搬已经写完的（对应 sweep 不再持有）
    if [ -s "$t" ] && ! fuser "$t" >/dev/null 2>&1; then
      n=$(wc -l < "$t")
      mv "$f" results/ && mv "$t" traces/ && \
        echo "[$(date '+%F %T')] 归档 ${f} + trace ${n} 行" >> $LOG
    fi
  done
  sleep 120
done
sleep 20
for f in budget_sweep_v2_Qwen3.6-27B-FP8_v55_*.json; do
  [ -e "$f" ] || continue
  t="tasks_sweep_Qwen3.6-27B-FP8_${f#budget_sweep_v2_Qwen3.6-27B-FP8_}"; t="${t%.json}.jsonl"
  mv "$f" results/ 2>/dev/null && echo "[$(date '+%F %T')] 收尾归档 $f" >> $LOG
  [ -s "$t" ] && mv "$t" traces/ 2>/dev/null
done
echo "[$(date '+%F %T')] 看守结束" >> $LOG
