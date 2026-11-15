#!/bin/bash
# SS-Pro with the 640-trained detector, tiled: tile 2 (~1280px tiles at 640 input) and tile 1 (~640px tiles).
SO=~/screen-options; PY=~/optbench/.venv/bin/python; LOG=~/relabel.log; SET=screenspot_pro
cd $SO; export PYTHONPATH=$SO/src
W=$SO/runs/det/axdet_640/weights/best.pt
for spec in "axdet640t2:2" "axdet640t1:1"; do
  NAME=${spec%%:*}; T=${spec#*:}; L=~/logs/chain-$NAME.log
  echo "$(date +%H:%M) $SET $NAME@640 detect start" >> $LOG
  $PY scripts/eval/relabel.py detect --set $SET --detector $NAME --weights $W --imgsz 640 --tile $T > $L 2>&1 || { echo "$NAME detect FAILED" >> $LOG; continue; }
  $PY scripts/eval/relabel.py variants --set $SET --detector $NAME --imgsz 640 --rlca ~/rlca >> $L 2>&1
  echo "$(date +%H:%M) $SET $NAME@640 lists: $(grep 'crop:' $L | sed 's/.*list //')" >> $LOG
  KEV_SLOTS=1 scripts/aws/kev_eval.sh $SO/data/relabel/$SET/$NAME@640/crop/records.jsonl relabel/$SET/$NAME@640/crop
done
echo "$(date +%H:%M) t640 done" >> $LOG
