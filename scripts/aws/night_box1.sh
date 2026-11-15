#!/bin/bash
# Box 1 overnight: score axdet_1280 (trained to its 3.5 h cap) it on the sandbox test apps, build its public-set lists.
set -u
SO=~/screen-options; PY=~/optbench/.venv/bin/python; LOG=~/relabel.log
cd $SO; export PYTHONPATH=$SO/src
W=runs/det/axdet_1280/weights
$PY scripts/eval/det_eval.py --data data/det --models $SO/$W/best.pt --sizes 640 1280 --out runs/det_eval_ours.json > ~/logs/det_eval_ours.log 2>&1
echo "$(date +%H:%M) det_eval axdet_1280 done" >> $LOG
for sz in 1280 640; do for s in osworld_g screenspot_pro screenspot_v2; do
  scripts/aws/det_chain.sh axdet1280 $SO/$W/best.pt $s $sz
done; done
L=$(ls data/relabel/*/axdet1280*/crop/records.jsonl)
$PY scripts/eval/frozen_pick.py --model Qwen/Qwen3-8B --records $L --base data/relabel --out runs/pick --batch 2 > ~/logs/pick_8b_box1.log 2>&1
echo "$(date +%H:%M) pick 8B box1 done" >> $LOG
echo "$(date +%H:%M) night_box1 done" >> $LOG
