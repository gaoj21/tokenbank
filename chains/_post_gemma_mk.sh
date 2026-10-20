#!/bin/bash
# After the Makora Gemma runs: apply the fixed closing call to empty L1 replies (FRAMES, MuSiQue, BBH; MATH is
# already patched), then judge FRAMES/MuSiQue with gpt-oss-120b (already serving locally).
cd /home/j0018/Desktop/tokenbank
log() { echo "[$(date '+%F %T')] $*"; }
true
G=traces/tasks_sweep_gemma-4-26B-A4B-mk
.venv/bin/python analysis/patch_empty_l1.py FRAMES ${G}_l1r_fr100.jsonl 2>&1 | tail -1
.venv/bin/python analysis/patch_empty_l1.py MUSIQUE ${G}_l1r_mu100.jsonl 2>&1 | tail -1
echo "BBH already patched per shard"
log "L1 patched"
curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-120b || { log "judge server not up"; exit 1; }
FILES=""; for t in l1r_fr100 l1r_mu100 gx4k_fr100 gx4k_mu100 gx4kr2_fr100 gx4kr2_mu100; do FILES="$FILES,gemma-4-26B-A4B-mk_$t"; done
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_gemma_mk.jsonl JUDGE_THREADS=256 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JUDGE_FILES=${FILES#,}
rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { log "JUDGE FAILED"; exit 1; }
log "GEMMA MK JUDGED"
