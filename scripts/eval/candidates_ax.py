"""Candidate lists on sandbox screens, labelled by AX: the free supervision for pre-training the ranker.

Same candidates as candidates.py (Vision OCR lines + detector boxes, merged as at serving) on the sandbox detection
images (data/det/images/<split>), but no instructions: a candidate is positive when it matches a visible actionable
AX element (data/det/labels_ax; centre inside the AX box or IoU > 0.3, not a container 4x its size). Use the val
and test splits only: the detector was trained on the train apps, so its confidence there is not honest.

Output looks like candidates.py's, with one pseudo-example per screen whose target_ids are all AX-matched candidates.

  python scripts/eval/candidates_ax.py --splits val test --name axgpa_t --weights runs/det/axgpa_1280/weights/best.pt
  -> data/cands/sandbox_<split>/<name>.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/eval"))
from relabel import match  # noqa: E402
from screen_options.sources import merge  # noqa: E402


def ax_boxes(path, W, H):
    out = []
    for l in open(path).read().splitlines():
        if l.strip():
            _, cx, cy, w, h = map(float, l.split())
            out.append([(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile", type=float, default=1)
    args = ap.parse_args()
    from PIL import Image
    from screen_options.detect import Detector
    det = Detector(model=args.weights, imgsz=args.imgsz, tile=args.tile)
    for split in args.splits:
        out = ROOT / "data/cands" / f"sandbox_{split}" / f"{args.name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for i, r in enumerate(map(json.loads, open(ROOT / "data/sandbox_ocr" / f"{split}.jsonl"))):
                rel = f"det/images/{split}/{r['id']}.jpg"
                img = Image.open(ROOT / "data" / rel).convert("RGB")
                W, H = img.size
                frame = (0, 0, W, H)
                items, _ = merge([], [{"box": tuple(l["box"]), "text": l["text"]} for l in r["lines"]],
                                 det.boxes(img, frame, 1.0), frame)
                els = [{"id": f"e{k}", "role": it.role, "box": list(it.box), "sources": sorted(it.sources),
                        "conf": round(it.conf, 4), "text": it.label if it.role == "text" else ""}
                       for k, it in enumerate(items)]
                pos = set()
                for b in ax_boxes(ROOT / "data/det/labels_ax" / split / f"{r['id']}.txt", W, H):
                    pos.update(match(els, gt_box=b)[:1])       # the best-matching candidate per AX element
                f.write(json.dumps({"image": rel, "size": [W, H], "elements": els,
                                    "examples": [{"id": r["id"], "instruction": "", "target_ids": sorted(pos)}]},
                                   ensure_ascii=False) + "\n")
                if i % 500 == 0:
                    print(f"sandbox_{split} {i}: {len(els)} candidates, {len(pos)} AX-matched", flush=True)


if __name__ == "__main__":
    main()
