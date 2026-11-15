"""Detectors against AX on held-out apps: how many of the AX elements each one boxes, and at what cost.

Test screens are the sandbox test-split apps (data/det/ax/images/test, never trained on). For each detector and
input size:
  mAP50, mAP50-95   ultralytics val, single class (every detector's classes merged), standard COCO-style
  recall@0.5        share of AX boxes matched by a prediction with IoU >= 0.5, at the serving settings
  center_recall     share of AX boxes whose centre lies inside some prediction (enough to click it)
  boxes             median predictions per screen at the serving settings (list length cost)
  ms                GPU inference per image at the serving settings
Serving settings are screen_options.detect.Detector's: conf 0.05 and each model's own NMS IoU.

  python scripts/eval/det_eval.py --data data/det --models gpa omni_v2 screen2ax runs/det/axdet_1280/weights/best.pt \
      --sizes 640 1280 --out runs/det_eval.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def iou_matrix(a, b):
    """a: (n,4) gt xyxy, b: (m,4) pred xyxy -> (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def gt_boxes(label_path, w, h):
    rows = [l.split() for l in open(label_path).read().splitlines() if l.strip()]
    out = [((float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h,
            (float(cx) + float(bw) / 2) * w, (float(cy) + float(bh) / 2) * h) for _, cx, cy, bw, bh in rows]
    return np.array(out).reshape(-1, 4)


def serving_metrics(det, images, labels_dir, limit):
    hit = chit = tot = 0
    nbox, ms = [], []
    for p in images[:limit]:
        im = Image.open(p).convert("RGB")
        gt = gt_boxes(labels_dir / (p.stem + ".txt"), im.width, im.height)
        t0 = time.perf_counter()
        pred = det.boxes(im, (0, 0, im.width, im.height), 1.0)
        ms.append((time.perf_counter() - t0) * 1000)
        pb = np.array([q["pixels"] for q in pred], dtype=float).reshape(-1, 4)
        nbox.append(len(pb))
        tot += len(gt)
        if len(gt) and len(pb):
            hit += int((iou_matrix(gt, pb).max(1) >= 0.5).sum())
            cx, cy = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2
            inside = (cx[:, None] >= pb[None, :, 0]) & (cx[:, None] <= pb[None, :, 2]) & \
                     (cy[:, None] >= pb[None, :, 1]) & (cy[:, None] <= pb[None, :, 3])
            chit += int(inside.any(1).sum())
    return {"recall@0.5": hit / tot, "center_recall": chit / tot, "boxes": statistics.median(nbox),
            "ms": statistics.median(ms[5:] or ms), "screens": len(nbox), "gt": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/det")
    ap.add_argument("--labels", default="ax")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 1280])
    ap.add_argument("--limit", type=int, default=100000, help="screens for the serving metrics")
    ap.add_argument("--out", default="runs/det_eval.json")
    args = ap.parse_args()
    from screen_options.detect import Detector
    root = (Path(args.data) / args.labels).resolve()
    yaml = Path(args.data).resolve() / f"{args.labels}.abs.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: element\n")
    images = sorted((root / "images/test").glob("*.jpg"))
    out_path = Path(args.out)
    results = json.load(open(out_path)) if out_path.exists() else []
    done = {(r["model"], r["imgsz"]) for r in results}
    for m in args.models:
        for sz in args.sizes:
            if (m, sz) in done:
                continue
            det = Detector(model=m, imgsz=sz)
            v = det.model.val(data=str(yaml), split="test", imgsz=sz, batch=8, single_cls=True, plots=False,
                              verbose=False, workers=4)
            r = {"model": m, "imgsz": sz, "mAP50": v.box.map50, "mAP50-95": v.box.map,
                 **serving_metrics(det, images, root / "labels/test", args.limit)}
            print(json.dumps(r), flush=True)
            results.append(r)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(results, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
