#!/bin/bash
# P2 收尾归档：等 sweep 进程退出后，把产物从项目根目录移到约定位置。
# 这个项目的 sweep 把 trace/json 写在根目录，靠人工移进 traces/ 和 results/
# （Ab6 那次就漏过一次）。脱离交互会话跑，保证过夜也能自动归位。
cd /home/j0018/Desktop/tokenbank
TRACE=tasks_sweep_gemma-4-31B-it_v54b_ma_fmt.jsonl
JSON=budget_sweep_v2_gemma-4-31B-it_v54b_ma_fmt.json
LOG=logs/_v54b_file_artifacts.log

echo "[$(date '+%F %T')] 等待 sweep 退出..." >> $LOG
while pgrep -f "^\.venv/bin/python -u _budget_sweep_v2" >/dev/null 2>&1; do sleep 60; done
sleep 10   # 让文件句柄关闭、缓冲落盘

if [ -f "$TRACE" ]; then
  n=$(wc -l < "$TRACE")
  mv "$TRACE" traces/ && echo "[$(date '+%F %T')] trace 已归档: $n 行 -> traces/$TRACE" >> $LOG
else
  echo "[$(date '+%F %T')] 警告: 未找到 $TRACE（run 可能失败）" >> $LOG
fi
[ -f "$JSON" ] && mv "$JSON" results/ && echo "[$(date '+%F %T')] json -> results/$JSON" >> $LOG

grep -E "SR=|^Exported:" logs/_v54b_math_gemma_fmt.log 2>/dev/null | tail -3 >> $LOG
echo "[$(date '+%F %T')] 归档完成" >> $LOG
