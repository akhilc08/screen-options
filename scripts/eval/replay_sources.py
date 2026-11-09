"""Replay an exploration run's saved screenshots through the pixel sources, one at a time, and time them.

Run after scripts/data/explore.py has finished, with nothing else busy, so the timings are clean.
For each unique screenshot: Vision OCR (accurate and fast), the OmniParser detector, the merge with
that state's AX elements, and captions for the detector boxes nothing else explains (captions only
on a --caption_sample of screenshots, since they cost seconds each). AX timings were recorded live.

Writes <run>/sources.jsonl (one row per screenshot) and <run>/sources_summary.json
(p50/p90 latency and item counts per app, and per target).

  .venv/bin/python scripts/eval/replay_sources.py --run data/explore/<stamp>
"""
import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import objc
import Quartz
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from screen_options.detect import Captioner, Detector  # noqa: E402
from screen_options.elements import Element  # noqa: E402
from screen_options.ocr import ocr_lines  # noqa: E402
from screen_options.sources import merge  # noqa: E402


def cgimage(path):
    src = Quartz.CGImageSourceCreateWithURL(Quartz.CFURLCreateWithFileSystemPath(None, str(path), 0, False), None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def ms(t0):
    return round((time.perf_counter() - t0) * 1000, 1)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--caption_sample", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run)
    states = {}
    for line in open(run / "states.jsonl"):
        s = json.loads(line)
        states.setdefault(s["shot"], s)        # first state that produced each screenshot
    shots = list(states)
    if args.limit:
        shots = shots[: args.limit]
    random.seed(0)
    caption_set = set(random.sample(shots, min(args.caption_sample, len(shots))))
    det, cap = Detector(), Captioner()
    if shots:                                   # first calls load models; keep them out of the timings
        s0 = states[shots[0]]
        warm = Image.open(run / shots[0]).convert("RGB")
        ocr_lines(cgimage(run / shots[0]), s0["frame"])
        ocr_lines(cgimage(run / shots[0]), s0["frame"], fast=True)
        det.boxes(warm, s0["frame"], s0["scale"])
        cap.captions(warm, [(0, 0, 64, 64)])
    rows = []
    with open(run / "sources.jsonl", "w") as f:
        for i, shot in enumerate(shots):
            # Vision/Quartz objects are autoreleased; without a pool per screenshot they pile up (29 GB once).
            with objc.autorelease_pool():
                s = states[shot]
                frame, scale = s["frame"], s["scale"]
                img_path = run / shot
                img = Image.open(img_path).convert("RGB")
                cg = cgimage(img_path)
                t0 = time.perf_counter()
                ocr = ocr_lines(cg, frame)
                t_ocr = ms(t0)
                t0 = time.perf_counter()
                ocr_fast = ocr_lines(cg, frame, fast=True)
                t_fast = ms(t0)
                t0 = time.perf_counter()
                boxes = det.boxes(img, frame, scale)
                t_det = ms(t0)
                ax = [Element(**{**e, "frame": tuple(e["frame"]) if e.get("frame") else None}) for e in s["elements"]]
                t0 = time.perf_counter()
                items, need = merge(ax, ocr, boxes, frame)
                t_merge = ms(t0)
                t_cap = None
                if shot in caption_set and need:
                    px = [(int((items[j].box[0] - frame[0]) * scale), int((items[j].box[1] - frame[1]) * scale),
                           int((items[j].box[0] - frame[0] + items[j].box[2]) * scale),
                           int((items[j].box[1] - frame[1] + items[j].box[3]) * scale)) for j in need]
                    t0 = time.perf_counter()
                    for j, text in zip(need, cap.captions(img, px)):
                        items[j].label = text
                    t_cap = ms(t0)
                row = {"shot": shot, "state": s["id"], "app": s["app"], "target": s["target"],
                       "ms": {**s.get("ms", {}), "ocr_accurate": t_ocr, "ocr_fast": t_fast, "detect": t_det,
                              "merge": t_merge, "caption": t_cap},
                       "counts": {"ax": len(ax), "ocr": len(ocr), "ocr_fast": len(ocr_fast), "det": len(boxes),
                                  "merged": len(items), "icons_uncaptioned": len(need)},
                       "items": [{"label": it.label, "role": it.role, "box": it.box, "sources": sorted(it.sources)}
                                 for it in items]}
                f.write(json.dumps(row) + "\n")
                rows.append(row)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if (i + 1) % 50 == 0:
                    print(f"{i + 1}/{len(shots)}", flush=True)
    summary = {}
    for key in ("app", "target"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        summary[key] = {}
        for g, rs in sorted(groups.items()):
            entry = {"screens": len(rs)}
            for m in ("capture", "ax_window", "ocr_accurate", "ocr_fast", "detect", "merge", "caption"):
                xs = [r["ms"][m] for r in rs if r["ms"].get(m) is not None]
                entry[f"{m}_p50"], entry[f"{m}_p90"] = pct(xs, 0.5), pct(xs, 0.9)
            for c in ("ax", "ocr", "det", "merged", "icons_uncaptioned"):
                entry[f"{c}_mean"] = round(statistics.mean(r["counts"][c] for r in rs), 1)
            summary[key][g] = entry
    (run / "sources_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["app"], indent=1))


if __name__ == "__main__":
    main()
