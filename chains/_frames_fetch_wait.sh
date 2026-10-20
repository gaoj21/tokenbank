#!/bin/bash
# Wait until Wikipedia stops rate-limiting this host (one light probe a minute), then fetch the FRAMES articles.
cd /home/j0018/Desktop/tokenbank
until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 30 -A 'climb-research/1.0 (academic research; batch of 50 titles per request)' 'https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&rvslots=main&titles=Harriet%20Lane&format=json&formatversion=2')" = 200 ]; do sleep 60; done
echo "[$(date '+%F %T')] rate limit lifted"
HF_DATASETS_OFFLINE=1 .venv/bin/python -u analysis/frames_fetch.py
