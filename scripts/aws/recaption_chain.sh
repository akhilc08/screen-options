#!/bin/bash
# Re-label one GPA list's icons with the AX-trained labeller, build variants, score Kev on crop and near_ctx.
#   scripts/aws/recaption_chain.sh <set> <imgsz>
set -u
SET=$1; SZ=$2
SO=~/screen-options; PY=~/optbench/.venv/bin/python; LOG=~/relabel.log
note() { echo "$(date +%H:%M) $SET gpa@$SZ+ax $*" >> $LOG; }
cd $SO
note "recaption start"
$PY scripts/eval/relabel.py recaption --set $SET --detector gpa --imgsz $SZ --captioner ~/.cache/rlca/ax-labeler --tag ax > ~/logs/recap-$SET-$SZ.log 2>&1 || { note "recaption FAILED"; exit 1; }
$PY scripts/eval/relabel.py variants --set $SET --detector gpa --imgsz $SZ --list ax --rlca ~/rlca >> ~/logs/recap-$SET-$SZ.log 2>&1 || { note "variants FAILED"; exit 1; }
note "variants done"
for v in crop near_ctx; do
  scripts/aws/kev_eval.sh $SO/data/relabel/$SET/gpa@$SZ+ax/$v/records.jsonl relabel/$SET/gpa@$SZ+ax/$v
done
note "all done"
