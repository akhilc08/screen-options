#!/bin/bash
# Score trained Kev (runs/sft-0.8b-v1) on one records file, at most KEV_SLOTS runs at a time on this box.
#   scripts/aws/kev_eval.sh <records.jsonl> <out name under ~/kev/runs>
set -u
DATA=$1; OUT=$2; SLOTS=${KEV_SLOTS:-2}; LOG=~/relabel.log
mkdir -p ~/logs ~/.kevlock
cd ~/kev
export KEV_PAD_BUCKET=256 KEV_EVAL_CONTEXT=serving
tag=$(echo "$OUT" | tr '/@' '__')
while :; do
  for i in $(seq 1 "$SLOTS"); do
    exec 9>~/.kevlock/slot$i
    if flock -n 9; then
      echo "$(date +%H:%M) kev $OUT start" >> $LOG
      .venv/bin/python -m kev.benchmark --device cuda --run runs/sft-0.8b-v1 --data "$DATA" --out "runs/$OUT" > ~/logs/kev-$tag.log 2>&1 \
        && echo "$(date +%H:%M) kev $OUT done" >> $LOG || echo "$(date +%H:%M) kev $OUT FAILED" >> $LOG
      exit 0
    fi
    exec 9>&-
  done
  sleep 20
done
