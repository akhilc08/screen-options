"""Option benchmark v2 (macOS): Apple Vision OCR, fast and accurate, over every screenshot in a v2 manifest.

Boxes come out in image pixels with the line text as the class, in the same row format optbench_detect.py writes:
{"key", "ms", "boxes": [[x, y, w, h, conf, text], ...]}. Runs headless (Vision needs no GUI session or TCC grant).

  python scripts/eval/optbench2_ocr.py --bench data/option_bench2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import objc
import Quartz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from screen_options.ocr import ocr_lines  # noqa: E402


def cgimage(path):
    src = Quartz.CGImageSourceCreateWithURL(Quartz.CFURLCreateWithFileSystemPath(None, str(path), 0, False), None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/option_bench2")
    args = ap.parse_args()
    bench = Path(args.bench)
    rows = [json.loads(l) for l in open(bench / "manifest.jsonl")]
    (bench / "sources").mkdir(exist_ok=True)
    outs = {m: open(bench / f"sources/ocr_{m}.jsonl", "w") for m in ("fast", "accurate")}
    for i, r in enumerate(rows):
        with objc.autorelease_pool():          # Vision objects pile up without a pool per image
            cg = cgimage((bench / r["image"]).resolve())
            frame = (0, 0, Quartz.CGImageGetWidth(cg), Quartz.CGImageGetHeight(cg))
            for mode, f in outs.items():
                t0 = time.perf_counter()
                lines = ocr_lines(cg, frame, fast=mode == "fast")
                ms = round((time.perf_counter() - t0) * 1000, 1)
                f.write(json.dumps({"key": r["key"], "ms": ms,
                                    "boxes": [list(l["box"]) + [l["conf"], l["text"]] for l in lines]}) + "\n")
        if i % 200 == 0:
            print(f"{i}/{len(rows)}", flush=True)
    for f in outs.values():
        f.close()


if __name__ == "__main__":
    main()
