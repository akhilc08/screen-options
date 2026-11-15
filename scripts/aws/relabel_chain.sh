#!/bin/bash
# One set x one detector size on a GPU box: detect + caption, build the label variants, then score Kev on each.
#   scripts/aws/relabel_chain.sh <set> <records_s name> <imgsz> [kev variants: all|none]
# Kev runs are limited to KEV_SLOTS at a time per box (flock), since several chains share one A10G.
set -u
SET=$1; REC=$2; SZ=$3; KEV=${4:-all}
SO=~/screen-options; PY=~/optbench/.venv/bin/python; LOG=~/relabel.log
mkdir -p ~/logs
note() { echo "$(date +%H:%M) $SET gpa@$SZ $*" >> $LOG; }
cd $SO
note "detect start"
$PY scripts/eval/relabel.py detect --set $SET --detector gpa --imgsz $SZ > ~/logs/detect-$SET-$SZ.log 2>&1 || { note "detect FAILED"; exit 1; }
$PY scripts/eval/relabel.py variants --set $SET --detector gpa --imgsz $SZ --rlca ~/rlca >> ~/logs/detect-$SET-$SZ.log 2>&1 || { note "variants FAILED"; exit 1; }
note "variants done: $(grep 'target in list' ~/logs/detect-$SET-$SZ.log | tail -1 | sed 's/.*list //')"
touch ~/relabel_ready_${SET}_$SZ
[ "$KEV" = none ] && exit 0
for v in crop near ctx near_ctx; do
  scripts/aws/kev_eval.sh $SO/data/relabel/$SET/gpa@$SZ/$v/records.jsonl relabel/$SET/gpa@$SZ/$v
done
note "all done"
