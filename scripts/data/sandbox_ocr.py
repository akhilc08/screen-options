"""Apple Vision OCR (accurate) for the sandbox detection images, so pixel-only candidate lists can be built on them.

Runs on a Mac (needs pyobjc Vision; rl-computer-agent's venv has it). Images are data/det/images/<split>/<id>.jpg
(window shots, longest side 1280); boxes come back in those image pixels.

  ../rl-computer-agent/.venv/bin/python scripts/data/sandbox_ocr.py --splits val test
  -> data/sandbox_ocr/<split>.jsonl  {"id", "size", "lines": [{"text", "box", "conf"}]}
"""
import argparse
import json
import sys
from pathlib import Path

import Quartz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from screen_options.ocr import ocr_lines  # noqa: E402


def cgimage(path):
    src = Quartz.CGImageSourceCreateWithURL(Quartz.CFURLCreateWithFileSystemPath(None, str(path), 0, False), None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()
    out_dir = ROOT / "data/sandbox_ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        out = out_dir / f"{split}.jsonl"
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
        imgs = sorted((ROOT / "data/det/images" / split).glob("*.jpg"))
        with open(out, "a") as f:
            for i, p in enumerate(x for x in imgs if x.stem not in done):
                cg = cgimage(p.resolve())
                w, h = Quartz.CGImageGetWidth(cg), Quartz.CGImageGetHeight(cg)
                lines = ocr_lines(cg, (0, 0, w, h))
                f.write(json.dumps({"id": p.stem, "size": [w, h], "lines": lines}, ensure_ascii=False) + "\n")
                if i % 500 == 0:
                    print(f"{split} {i}/{len(imgs)}", flush=True)


if __name__ == "__main__":
    main()
