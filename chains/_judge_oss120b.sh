#!/bin/bash
# Judge gpt-oss-120b's FRAMES/MuSiQue answers with gpt-oss-120b (medium effort, temperature 0).
cd /home/j0018/Desktop/tokenbank
export JUDGE_MODEL=openai/gpt-oss-120b JUDGE_ENDPOINTS=http://127.0.0.1:18101/v1,http://127.0.0.1:18103/v1,http://127.0.0.1:18105/v1,http://127.0.0.1:18107/v1
export JUDGE_EXTRA_JSON='{"reasoning_effort":"medium"}' JUDGE_TEMPERATURE=0 JUDGE_CACHE=results/judge_cache_oss120b.jsonl
export JUDGE_QUEUE=results/judge_queue_oss120b_120.jsonl JUDGE_THREADS=64 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JUDGE_FILES=gpt-oss-120b_climbx_fr100,gpt-oss-120b_climbx_mu100,gpt-oss-120b_gx4k_fr100,gpt-oss-120b_gx4k_mu100
.venv/bin/python analysis/llm_judge.py collect
.venv/bin/python analysis/llm_judge.py run 0
.venv/bin/python analysis/llm_judge.py apply
echo "[$(date '+%F %T')] JUDGE 120B DONE"
