#!/bin/bash
# DeepSeek AgentPrune, MATH numeric subset only (GSM8K dropped 2026-10-03: on GLM its truncation
# rate at cap 1000 was 1-4% and the cap had no effect, so the boundary case is shown once).
cd /home/j0018/Desktop/tokenbank/external/AgentPrune
URL=http://127.0.0.1:18000/v1; MODEL=deepseek-v4.1-flash; OUT=../../results/agentprune/ds
# the cap-1000 Debate run lost its wrapper shell; wait for its python process to exit, then
# mark it done if it printed a final accuracy
while ps -eo args | grep -q "[r]un_gsm8k.py --llm_name deepseek-v4.1-flash --mode Debate"; do sleep 60; done
grep -q "Accuracy" $OUT/ds_math_numeric_Debate_cap1000.log && ! grep -q DONE $OUT/ds_math_numeric_Debate_cap1000.log && echo DONE >> $OUT/ds_math_numeric_Debate_cap1000.log
run() { local mode=$1 n=$2 rounds=$3 cap=$4; local tag=ds_math_numeric_${mode}_cap${cap}
  grep -q DONE $OUT/$tag.log 2>/dev/null && { echo "SKIP $tag"; return; }
  rm -f $OUT/calls_$tag.jsonl; echo "[$(date '+%F %T')] START $tag"
  env BASE_URL=$URL API_KEY=EMPTY AP_MAX_TOKENS=$cap AP_TEMPERATURE=0 AP_CALL_LOG=$OUT/calls_$tag.jsonl AP_TAG=$tag \
    ../../.venv/bin/python experiments/run_gsm8k.py --llm_name $MODEL --mode $mode --agent_nums $n \
    --batch_size 32 --num_rounds $rounds --dataset_json dataset/gsm8k/math_numeric.jsonl > $OUT/$tag.log 2>&1 && echo DONE >> $OUT/$tag.log
  echo "[$(date '+%F %T')] DONE  $tag $(grep -oE 'Accuracy: [0-9.]+' $OUT/$tag.log | tail -1)"; }
for cap in 4096 none; do run DirectAnswer 1 1 $cap & run FullConnected 4 1 $cap & run Debate 4 2 $cap & wait; done
echo "[$(date '+%F %T')] AGENTPRUNE DS DONE"
