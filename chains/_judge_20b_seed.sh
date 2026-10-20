#!/bin/bash
# Judge gpt-oss-20b FRAMES/MuSiQue (fixed reruns and the verify-seeding test) with gpt-oss-120b.
cd /home/j0018/Desktop/tokenbank
K=/home/j0018/Desktop/remote/id_rsa
for i in $(seq 1 240); do curl -s -m 5 http://127.0.0.1:18101/v1/models | grep -q gpt-oss-120b && break; sleep 15; done
echo "[$(date '+%F %T')] judge server ready"
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=$(for p in 18101 18102 18103 18104 18105 18106 18107 18108; do echo -n "http://127.0.0.1:$p/v1,"; done | sed 's/,$//')
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_20bseed.jsonl JUDGE_THREADS=256 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JUDGE_FILES=gpt-oss-20b_climbx_fr100,gpt-oss-20b_climbx_mu100,gpt-oss-20b_gx4k_fr100,gpt-oss-20b_gx4k_mu100,gpt-oss-20b_gx4kr2_fr100,gpt-oss-20b_gx4kr2_mu100,gpt-oss-20b_seedv_fr100,gpt-oss-20b_seedv_mu0100
rm -f $JUDGE_QUEUE
.venv/bin/python analysis/llm_judge.py collect && .venv/bin/python analysis/llm_judge.py run 0 && .venv/bin/python analysis/llm_judge.py apply || { echo "JUDGE FAILED"; exit 1; }
echo "[$(date '+%F %T')] JUDGE 20B SEED DONE"
