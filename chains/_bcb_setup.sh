#!/bin/bash
cd /home/j0018/Desktop/tokenbank
while pgrep -f "docker pull bigcodebench" >/dev/null; do sleep 30; done
docker images bigcodebench/bigcodebench-evaluate -q | grep -q . || { echo "IMAGE MISSING"; exit 1; }
echo "[$(date '+%F %T')] image ready, validating judge against canonical solutions"
N=60 .venv/bin/python -u _bcb_validate.py > logs/_bcb_validate.log 2>&1
echo "[$(date '+%F %T')] DONE bcb_validate"
