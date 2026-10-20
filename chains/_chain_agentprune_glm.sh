#!/bin/bash
# AgentPrune (ICLR 2025, released code) as a second, external multi-agent topology.
# Their single-agent baseline (DirectAnswer: one solver + decision node), FullConnected (4 solvers
# + decision node, 1 round) and Debate (4 solvers, 2 rounds, temporal edges), at three per-call
# caps: 1000 (the default declared in their llm.py), 4096 (our main setting) and none (what
# their released achat() actually does: it passes no max_tokens). Temperature 0 throughout.
# Datasets: their GSM8K (1,319) and the 320 MATH-500 items with numeric answers, fed through
# their GSM8K domain unchanged. Calls are logged per call with role and finish reason.
cd /home/j0018/Desktop/tokenbank/external/AgentPrune
URL=https://llm57.jierungogogo.com/v1; MODEL=zai-org/GLM-5.3-Flash; OUT=../../results/agentprune/glm; mkdir -p $OUT
run() { local ds=$1 mode=$2 n=$3 rounds=$4 cap=$5; local tag=glm_${ds}_${mode}_cap${cap}
  [ -s $OUT/calls_$tag.jsonl ] && grep -q DONE $OUT/$tag.log 2>/dev/null && { echo "SKIP $tag"; return; }
  rm -f $OUT/calls_$tag.jsonl
  echo "[$(date '+%F %T')] START $tag"
  env BASE_URL=$URL API_KEY=EMPTY AP_MAX_TOKENS=$cap AP_TEMPERATURE=0 AP_CALL_LOG=$OUT/calls_$tag.jsonl AP_TAG=$tag \
    ../../.venv/bin/python experiments/run_gsm8k.py --llm_name $MODEL --mode $mode --agent_nums $n \
    --batch_size 16 --num_rounds $rounds --dataset_json dataset/gsm8k/$ds.jsonl > $OUT/$tag.log 2>&1 && echo DONE >> $OUT/$tag.log
  echo "[$(date '+%F %T')] DONE  $tag $(grep -oE 'Accuracy: [0-9.]+' $OUT/$tag.log | tail -1)"; }
for ds in math_numeric gsm8k; do
  for cap in 1000 4096 none; do
    run $ds DirectAnswer 1 1 $cap &
    run $ds FullConnected 4 1 $cap &
    run $ds Debate 4 2 $cap &
    wait
  done
done
echo "[$(date '+%F %T')] AGENTPRUNE GLM DONE"
