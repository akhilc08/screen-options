"""YOLO detection set from the sandbox screens: AX boxes are the labels, so the detector learns from free supervision.

Each sandbox screen is a window screenshot (frame in screen points, `scale` pixels per point) plus the AX
elements visible in it (actionable roles only: links, buttons, rows, fields, ...). Screens flagged ax_poor or
left out of every split: their missing AX would teach the detector to ignore real controls, and would
score real detections as false positives. Screens flagged capture_mismatch stay in: that flag only means many AX
elements are off the image (below the fold, separate popover windows), and those are already dropped by
"visible". Splits are by app (data/sandbox_ds/splits.json), so test apps are never trained on.

Two label sets share one image folder:
  ax       visible AX boxes only
  ax+gpa   plus GPA boxes (conf >= --gpa_conf) that overlap no AX box (IoU < 0.3), to fill what AX leaves out

Images are resized so the longest side is --long (default 1280) and saved as JPEG.

  python3 scripts/data/det_dataset.py [--workers 8]
  -> data/det/images/{train,val,test}/<id>.jpg, data/det/labels_{ax,ax+gpa}/{train,val,test}/<id>.txt,
     data/det/{ax,ax+gpa}.yaml (relative paths; the folder can be rsynced as is)
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/sandbox_ds"
OUT = ROOT / "data/det"
MIN_SIDE = 4          # points


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    w = min(ax2, bx2) - max(a[0], b[0])
    h = min(ay2, by2) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def yolo_line(box, frame):
    """Screen-point box -> 'cls cx cy w h' normalised to the window, clipped; None if too small."""
    fx, fy, fw, fh = frame
    x1, y1 = max(box[0], fx), max(box[1], fy)
    x2, y2 = min(box[0] + box[2], fx + fw), min(box[1] + box[3], fy + fh)
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return f"0 {((x1 + x2) / 2 - fx) / fw:.6f} {((y1 + y2) / 2 - fy) / fh:.6f} {(x2 - x1) / fw:.6f} {(y2 - y1) / fh:.6f}"


def one(job):
    s, o, split, long_side, gpa_conf = job
    img_out = OUT / "images" / split / f"{s['id']}.jpg"
    if not img_out.exists():
        shot = ROOT / s["shot"]
        if not shot.exists():
            return None
        im = Image.open(shot).convert("RGB")
        r = long_side / max(im.size)
        if r < 1:
            im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
        img_out.parent.mkdir(parents=True, exist_ok=True)
        im.save(img_out, quality=90)
    frame = s["frame"]
    ax = [e["box"] for e in o["options"] if e["id"].startswith("a") and e.get("visible")]
    extra = [g["box"] for g in o.get("gpa", []) if g["conf"] >= gpa_conf and all(iou(g["box"], a) < 0.3 for a in ax)]
    n = {}
    for name, boxes in (("ax", ax), ("ax+gpa", ax + extra)):
        lines = [l for l in (yolo_line(b, frame) for b in boxes) if l]
        p = OUT / f"labels_{name}" / split / f"{s['id']}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + ("\n" if lines else ""))
        n[name] = len(lines)
    return split, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--long", type=int, default=1280)
    ap.add_argument("--gpa_conf", type=float, default=0.5)
    args = ap.parse_args()
    apps = json.load(open(DS / "splits.json"))["apps"]
    screens = {s["id"]: s for s in map(json.loads, open(DS / "screens.jsonl"))}
    jobs = []
    for line in open(DS / "options.jsonl"):
        o = json.loads(line)
        s = screens[o["id"]]
        split = apps[s["app"]]
        if s.get("ax_poor"):
            continue
        jobs.append((s, o, split, args.long, args.gpa_conf))
    tot = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=32)):
            if r is None:
                continue
            split, n = r
            t = tot.setdefault(split, {"screens": 0, "ax": 0, "ax+gpa": 0})
            t["screens"] += 1
            for k, v in n.items():
                t[k] += v
            if i % 2000 == 0:
                print(i, len(jobs), flush=True)
    for name in ("ax", "ax+gpa"):
        # Ultralytics finds labels by swapping "images" for "labels" in the (symlink-resolved) image path, so each
        # label set gets its own root holding hard links to the shared images and a copy of its labels.
        root = OUT / name
        if not (root / "images").exists():
            shutil.copytree(OUT / "images", root / "images", copy_function=os.link)
        shutil.rmtree(root / "labels", ignore_errors=True)
        shutil.copytree(OUT / f"labels_{name}", root / "labels")
        (OUT / f"{name}.yaml").write_text(
            f"path: {name}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: element\n")
    json.dump(tot, open(OUT / "stats.json", "w"), indent=1)
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
