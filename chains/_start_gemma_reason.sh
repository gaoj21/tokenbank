#!/bin/bash
# Gemma-4-31B-it (bf16) with thinking, on the remote H20s: wait for the weights to finish
# downloading and verifying, start vLLM with the gemma4 reasoning parser (thinking returned
# separately from the answer), and open the local tunnel on 18000.
cd /home/j0018/Desktop/tokenbank
KEY=/home/j0018/Desktop/remote/id_rsa; RS="ssh -i $KEY -p 820 -o ConnectTimeout=15 -o BatchMode=yes root@47.82.65.107"
until $RS 'cd /mnt/data/models/gemma-4-31b-it && [ ! -e model-00001-of-00002.safetensors.parallel ] && [ ! -e model-00002-of-00002.safetensors.parallel ] && [ -e model-00001-of-00002.safetensors ] && [ -e model-00002-of-00002.safetensors ] && ! pgrep -f "[m]s_fetch.py" >/dev/null'; do sleep 60; done
echo "[$(date '+%F %T')] weights complete"; $RS 'tail -3 /mnt/data/models/gemma4_31b_fetch2.log'
$RS 'cd /mnt/data/models && setsid nohup bash serve_gemma4_31b_reason.sh > serve_gemma4_31b_reason.log 2>&1 < /dev/null & echo started'
pgrep -f "18000:127.0.0.1:8000" >/dev/null || setsid nohup ssh -i $KEY -p 820 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L 18000:127.0.0.1:8000 root@47.82.65.107 > logs/_tunnel_gemma.log 2>&1 < /dev/null &
until curl -s -m 5 http://127.0.0.1:18000/v1/models | grep -q gemma-4-31b-it; do sleep 30; done
echo "[$(date '+%F %T')] GEMMA REASON READY"
