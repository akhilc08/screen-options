#!/bin/bash
# Tiled detection on one public set: our detector and GPA, both at 1280 with ~1280px tiles.
#   scripts/aws/tile_chain.sh [set]   (default screenspot_pro)
set -u
SET=${1:-screenspot_pro}
SO=~/screen-options; PY=~/optbench/.venv/bin/python; LOG=~/relabel.log
cd $SO; export PYTHONPATH=$SO/src
while pgrep -f "[f]rozen_pick.py" > /dev/null; do sleep 30; done
for spec in ${SPECS:-"axdet1280t:$SO/runs/det/axdet_1280/weights/best.pt" "gpat:gpa"}; do
  NAME=${spec%%:*}; W=${spec#*:}
  L=~/logs/chain-$NAME.log
  echo "$(date +%H:%M) $SET $NAME@1280 detect start" >> $LOG
  $PY scripts/eval/relabel.py detect --set $SET --detector $NAME --weights $W --imgsz 1280 --tile 1 > $L 2>&1 || { echo "$(date +%H:%M) $NAME detect FAILED" >> $LOG; continue; }
  $PY scripts/eval/relabel.py variants --set $SET --detector $NAME --imgsz 1280 --rlca ~/rlca >> $L 2>&1
  echo "$(date +%H:%M) $SET $NAME@1280 lists: $(grep 'crop:' $L | sed 's/.*list //')" >> $LOG
  KEV_SLOTS=1 scripts/aws/kev_eval.sh $SO/data/relabel/$SET/$NAME@1280/crop/records.jsonl relabel/$SET/$NAME@1280/crop
  $PY scripts/eval/frozen_pick.py --model Qwen/Qwen3-8B --records data/relabel/$SET/$NAME@1280/crop/records.jsonl --base data/relabel --out runs/pick --batch 2 >> ~/logs/pick_8b_tile.log 2>&1
done
echo "$(date +%H:%M) tile_chain done" >> $LOG
