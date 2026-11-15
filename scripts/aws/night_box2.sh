#!/bin/bash
# Box 2 overnight: a 640px detector (the fast setting) and an AX+GPA-label ablation at 1280, each scored on the
# sandbox test apps and turned into public-set lists; then frozen Qwen3-8B on every list this box built.
set -u
SO=~/screen-options; PY=~/ob2/.venv/bin/python; LOG=~/relabel.log
cd $SO; export PYTHONPATH=$SO/src
note() { echo "$(date +%H:%M) $*" >> $LOG; }
note "train axdet_640 start"
$PY scripts/train/train_detector.py --data data/det --labels ax --imgsz 640 --epochs 15 --batch 16 --hours 2 --name axdet_640 > ~/logs/train_axdet_640.log 2>&1
note "train axdet_640 done"
$PY scripts/eval/det_eval.py --data data/det --models $SO/runs/det/axdet_640/weights/best.pt --sizes 640 --out runs/det_eval_ours.json > ~/logs/det_eval_640.log 2>&1
for s in osworld_g screenspot_pro screenspot_v2; do scripts/aws/det_chain.sh axdet640 $SO/runs/det/axdet_640/weights/best.pt $s 640; done
note "train axgpa_1280 start"
$PY scripts/train/train_detector.py --data data/det --labels ax+gpa --imgsz 1280 --epochs 15 --batch 8 --hours 3.5 --name axgpa_1280 > ~/logs/train_axgpa_1280.log 2>&1
note "train axgpa_1280 done"
$PY scripts/eval/det_eval.py --data data/det --models $SO/runs/det/axgpa_1280/weights/best.pt --sizes 640 1280 --out runs/det_eval_ours.json > ~/logs/det_eval_axgpa.log 2>&1
for s in osworld_g screenspot_pro screenspot_v2; do scripts/aws/det_chain.sh axgpa1280 $SO/runs/det/axgpa_1280/weights/best.pt $s 1280; done
L=$(ls data/relabel/*/axdet640*/crop/records.jsonl data/relabel/*/axgpa1280*/crop/records.jsonl)
$PY scripts/eval/frozen_pick.py --model Qwen/Qwen3-8B --records $L --base data/relabel --out runs/pick --batch 2 > ~/logs/pick_8b_box2.log 2>&1
note "night_box2 done"
