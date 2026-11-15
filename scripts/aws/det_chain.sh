#!/bin/bash
# Option lists from one of our detectors on one public set, then Kev on them.
#   scripts/aws/det_chain.sh <name> <weights.pt> <set> <imgsz> [kev: yes|no]
# Builds <name>@<imgsz> (detector + OmniParser captions) and <name>@<imgsz>+ax (icons re-labelled by the
# AX-trained labeller), the label variants of each, and scores Kev on the crop variant of both.
set -u
NAME=$1; W=$2; SET=$3; SZ=$4; KEV=${5:-yes}
SO=~/screen-options; PY=~/optbench/.venv/bin/python; LOG=~/relabel.log
[ -x $PY ] || PY=~/ob2/.venv/bin/python
mkdir -p ~/logs
L=~/logs/chain-$NAME-$SET-$SZ.log
note() { echo "$(date +%H:%M) $SET $NAME@$SZ $*" >> $LOG; }
cd $SO
export PYTHONPATH=$SO/src
note "detect start"
$PY scripts/eval/relabel.py detect --set $SET --detector $NAME --weights $W --imgsz $SZ > $L 2>&1 || { note "detect FAILED"; exit 1; }
$PY scripts/eval/relabel.py recaption --set $SET --detector $NAME --imgsz $SZ --captioner ~/.cache/rlca/ax-labeler --tag ax >> $L 2>&1 || { note "recaption FAILED"; exit 1; }
$PY scripts/eval/relabel.py variants --set $SET --detector $NAME --imgsz $SZ --rlca ~/rlca >> $L 2>&1 || { note "variants FAILED"; exit 1; }
$PY scripts/eval/relabel.py variants --set $SET --detector $NAME --imgsz $SZ --list ax --rlca ~/rlca >> $L 2>&1 || { note "variants ax FAILED"; exit 1; }
note "lists done: $(grep 'crop:' $L | sed 's/.*list //' | tr '\n' ' ')"
[ "$KEV" = no ] && exit 0
for l in $NAME@$SZ $NAME@$SZ+ax; do
  KEV_SLOTS=1 scripts/aws/kev_eval.sh $SO/data/relabel/$SET/$l/crop/records.jsonl relabel/$SET/$l/crop
done
note "all done"
