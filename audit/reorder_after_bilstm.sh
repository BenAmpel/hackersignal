#!/bin/bash
set -u
W="/Volumes/Extreme SSD/_worktrees/exploit-text-graph/camera-ready-audit"
J="$W/results/camera_ready/raw/task2.jsonl"
until [ "$(grep -c '"model": "BiLSTM", "condition": "temporal_ood", "variant": "packed_last_token"' "$J")" -ge 3 ]; do sleep 30; done
echo "corrected BiLSTM complete $(date); stopping old queue"
for p in $(pgrep -f "launch_task2.sh"); do kill "$p"; done
for p in $(pgrep -f "run_task2.py --stage rnn"); do kill "$p"; done
sleep 5
mv "$W/logs/task2.log" "$W/logs/task2_part3.log"
nohup "$W/audit/launch_task2_v2.sh" > "$W/logs/task2.log" 2>&1 < /dev/null &
echo "new queue started $(date)"
