"""Candidate lists before any captioning: OCR text + detector boxes merged as at serving, with what a ranker may use.

For each screen: every candidate's box, role (text / icon), sources, detector confidence and OCR text (free, it's
already read), plus the screen's examples with the candidates that match the target (relabel.match, uncapped).
Rankers (scripts/eval/rank.py) order these lists; only icons kept in the top K would then be captioned.

Eval sets use the fixed 500-example samples (as relabel.py); other sets (training data for learned rankers) use
every example, up to --max_screens screens.

  python scripts/eval/candidates.py --set screenspot_pro --name axgpa --weights runs/det/axgpa_1280/weights/best.pt \
      --imgsz 1280 --tile 1
  -> data/cands/<set>/<name>.jsonl
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/eval"))
from relabel import DATA, PUBLIC_ACTIONS, RECORDS_S, match, sample_examples  # noqa: E402
from screen_options.sources import merge  # noqa: E402


def subpoints(els, W, H, side=8):
    """Points inside widgets that a whole-element list can't express: positions along bars (sliders, scrollbars,
    progress) and the corners / edge midpoints of large boxes (resize grips, scroll edges). Each is a small
    side x side candidate with role "point"; the instruction ranker scores it like any other."""
    out = []
    def add(px, py, src):
        px, py = min(max(px, side / 2), W - side / 2), min(max(py, side / 2), H - side / 2)
        out.append({"role": "point", "box": [round(px - side / 2), round(py - side / 2), side, side],
                    "sources": [src], "conf": 0.0, "text": ""})
    for e in els:
        x, y, w, h = e["box"]
        if e["role"] == "text":
            continue
        if w >= 4 * h and w >= 60:          # horizontal bar
            for f in (0.1, 0.25, 0.5, 0.75, 0.9):
                add(x + f * w, y + h / 2, "bar")
        elif h >= 4 * w and h >= 60:        # vertical bar
            for f in (0.1, 0.25, 0.5, 0.75, 0.9):
                add(x + w / 2, y + f * h, "bar")
        if w >= 80 and h >= 80:             # panel: corners and edge midpoints, just inside
            i = 3
            for px, py in ((x + i, y + i), (x + w - i, y + i), (x + i, y + h - i), (x + w - i, y + h - i),
                           (x + w / 2, y + i), (x + w / 2, y + h - i), (x + i, y + h / 2), (x + w - i, y + h / 2)):
                add(px, py, "edge")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True, help="WEIGHTS name or .pt path")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile", type=float, default=0)
    ap.add_argument("--max_screens", type=int, default=1500)
    ap.add_argument("--conf", type=float, default=0.05, help="detector confidence threshold")
    ap.add_argument("--subpoints", action="store_true", help="add sub-element points (see subpoints)")
    ap.add_argument("--extra", help="a second detector (WEIGHTS name or .pt) whose boxes are added before merging")
    args = ap.parse_args()
    from PIL import Image
    from screen_options.detect import Detector

    if args.set in RECORDS_S and args.set not in ("mmbench_gui", "omniact"):
        examples = sample_examples(args.set)
    else:
        examples = list(map(json.loads, open(DATA / "converted" / args.set / "examples.jsonl")))
    examples = [e for e in examples if PUBLIC_ACTIONS.get(e.get("action")) and (e.get("gt_box") or e.get("gt_point"))]
    by_img = {}
    for e in examples:
        by_img.setdefault(e["image"], []).append(e)
    images = sorted(by_img)
    random.Random(0).shuffle(images)
    images = images[:args.max_screens]
    old = {s["image"]: s for s in map(json.loads, open(DATA / "converted" / args.set / "screens.jsonl"))}
    out = DATA / "cands" / args.set / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["image"] for l in open(out)} if out.exists() else set()
    det = Detector(model=args.weights, imgsz=args.imgsz, tile=args.tile, conf=args.conf)
    extra = Detector(model=args.extra, imgsz=args.imgsz, tile=args.tile, conf=args.conf) if args.extra else None
    print(f"{args.set}: {len(images)} screens, {len(done)} done", flush=True)
    with open(out, "a") as f:
        for i, rel in enumerate(r for r in images if r not in done and r in old and (DATA / "public" / r).exists()):
            img = Image.open(DATA / "public" / rel).convert("RGB")
            frame = (0, 0, img.width, img.height)
            ocr = [{"box": tuple(e["box"]), "text": e["label"]} for e in old[rel]["elements"] if "ocr" in e["sources"]]
            t0 = time.perf_counter()
            boxes = det.boxes(img, frame, 1.0)
            if extra:      # our boxes first, so merge keeps them where both detectors agree
                boxes += [{**b, "conf": b["conf"] * 0.5} for b in extra.boxes(img, frame, 1.0)]
            ms = (time.perf_counter() - t0) * 1000
            items, _ = merge([], ocr, boxes, frame)
            els = [{"id": f"e{k}", "role": it.role, "box": list(it.box), "sources": sorted(it.sources),
                    "conf": round(it.conf, 4), "text": it.label if it.role == "text" else ""}
                   for k, it in enumerate(items)]
            if args.subpoints:
                els += [{"id": f"e{len(els) + k}", **p} for k, p in enumerate(subpoints(els, img.width, img.height))]
            exs = [{"id": e["id"], "instruction": e["instruction"], "split": e.get("split"),
                    "target_ids": match(els, gt_box=e.get("gt_box"),
                                        gt_point=None if e.get("gt_box") else e.get("gt_point"))}
                   for e in by_img[rel]]
            f.write(json.dumps({"image": rel, "size": [img.width, img.height], "detect_ms": round(ms),
                                "elements": els, "examples": exs}, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"{args.set} {i}: {len(els)} candidates, {ms:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
