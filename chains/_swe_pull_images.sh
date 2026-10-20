#!/bin/bash
# Docker Hub's pull limit is 100/hour anonymous, 200/hour authenticated, and SWE-bench
# Verified needs 500 instance images. The evaluation harness treats a throttled
# pull as an instance error, so 402 of 500 "failed" for a reason that has
# nothing to do with the patches. Pulling separately, and sleeping through the
# limit, decouples the two: the harness then runs entirely from local cache.
# Already-present images are skipped, so this is safe to re-run at any time.
cd /home/j0018/Desktop/tokenbank
while :; do
  missing=0; pulled=0
  while read -r img; do
    [ -z "$img" ] && continue
    docker image inspect "$img" >/dev/null 2>&1 && continue
    missing=$((missing+1))
    if docker pull -q "$img" >/dev/null 2>&1; then
      pulled=$((pulled+1))
    else
      echo "[$(date '+%F %T')] throttled after $pulled pulls this round; $missing still missing"
      break
    fi
  done < swe_images.txt
  have=$(docker images --filter=reference="swebench/*" -q | wc -l)
  echo "[$(date '+%F %T')] local=$have/500 pulled_this_round=$pulled"
  [ "$have" -ge 500 ] && { echo "[$(date '+%F %T')] ALL IMAGES PRESENT"; break; }
  # The limit is 200 pulls per HOUR on a free authenticated account
  # (ratelimit-limit: 200;w=3600), not per six hours. Sleeping six hours
  # here wasted five of them; sleep just past the one-hour window.
  sleep 3660
done
