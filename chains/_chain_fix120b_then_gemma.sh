#!/bin/bash
# The gpt-oss-120b open-book runs used the default client timeout (600 s, two silent SDK resends, four
# retries), and under full load some calls exhausted every retry and were dropped. After that chain ends:
# rerun the affected tasks with LLM_TIMEOUT=3600, no SDK resends and up to 60 retries, judge the reruns on
# their own (fresh files, so the judge's backup cannot overwrite them), swap them in, re-export, recompute;
# then hand over to the Gemma chain.
cd /home/j0018/Desktop/tokenbank
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "OPENBOOK 120B ALL DONE\|JUDGE FAILED\|SERVER FAILED" logs/_chain_openbook_120b.log; do sleep 60; done
grep -q "OPENBOOK 120B ALL DONE" logs/_chain_openbook_120b.log || { log "120b chain failed"; exit 1; }
T=traces/tasks_sweep_gpt-oss-120b; D=results/climb_live/fix120b
.venv/bin/python analysis/fix_dropped_calls.py find $D ${T}_{l1rob,gx4kob,gx4kobr2}_{fr,mu}100.jsonl
export MODEL=gpt-oss-120b HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
B="BUDGETS=10000000 SPLIT=test L01_MAX_TOKENS=8192 OPEN_BOOK=1 LLM_TIMEOUT=3600 LLM_SDK_RETRIES=0 LLM_MAX_RETRIES=60"
C="CONFIG_ONLY=CLIMB-Live THETA_LIVE=99 V5_MAX_LEVEL=1 L1_MODE=reason"; F="CONFIG_ONLY=V5-Unc-Fmt"
p=18111; RUNS=""
for c in l1rob gx4kob gx4kobr2; do for b in fr mu; do
    ids=$D/tasks_sweep_gpt-oss-120b_${c}_${b}100.json
    [ "$(.venv/bin/python -c "import json; print(len(json.load(open('$ids'))))")" = 0 ] && continue
    cfg=$F; [ $c = l1rob ] && cfg=$C
    bench="BENCHMARK=FRAMES"; [ $b = mu ] && bench="BENCHMARK=MUSIQUE MUSIQUE_N=99999"
    PORT=$p MAX_WORKERS=96 bash chains/_gptoss_run.sh _${c}fix_${b}100 $bench $B $cfg TASK_IDS_FILE=$ids &
    RUNS="$RUNS ${c}:${b}"; p=$((p+1))
done; done
wait
log "reruns done:$RUNS"
FILES=""; for x in $RUNS; do FILES="$FILES,gpt-oss-120b_${x%%:*}fix_${x##*:}100"; done
if [ -n "$FILES" ]; then
    export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for q in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$q/v1,"; done | sed 's/,$//')
    export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
    export JUDGE_QUEUE=results/judge_queue_oss120b_fix120b.jsonl JUDGE_THREADS=256 JUDGE_FILES=${FILES#,}
    rm -f $JUDGE_QUEUE
    .venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { log "FIX JUDGE FAILED"; exit 1; }
    for x in $RUNS; do c=${x%%:*}; b=${x##*:}; .venv/bin/python analysis/fix_dropped_calls.py merge ${T}_${c}_${b}100.jsonl ${T}_${c}fix_${b}100.jsonl; done
fi
.venv/bin/python analysis/fix_dropped_calls.py find $D/after ${T}_{l1rob,gx4kob,gx4kobr2}_{fr,mu}100.jsonl
RECORDS_OUT=paper/data/records_ob120b.csv SUFFIX=100 .venv/bin/python analysis/export_records_new.py gpt-oss-120b:gpt-oss-120b:l1rob:gx4kob
(cd paper && CLIMB_TAG=l1rob REPEAT_TAG=gx4kobr2 MAIN_OUT=data/main_results_ob120b.json ../.venv/bin/python analysis/main_results.py data/records_ob120b.csv 2>&1 | grep -v Warn)
log "FIX 120B DONE"
bash chains/_chain_openbook_gemma.sh
