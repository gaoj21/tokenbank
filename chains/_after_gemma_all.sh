#!/bin/bash
# After the Gemma split-judge chain ends (FRAMES repeat judged): every analysis once more, with the
# case-insensitive answer extraction for the agreement signal.
cd /home/j0018/Desktop/tokenbank
until grep -q "GEMMA ALL DONE\|FAILED" logs/_chain_gemma_split_judge.log; do sleep 60; done
grep -q "GEMMA ALL DONE" logs/_chain_gemma_split_judge.log || { echo "gemma chain failed"; exit 1; }
bash chains/_rerun_analyses_ob.sh
