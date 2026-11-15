"""Detection latency per screen on a public set, one detector config at a time on an otherwise idle GPU.

Config: name:model:imgsz[:tile], model a WEIGHTS name or a .pt path. Reports median / p90 ms per screen
(CUDA-synchronised, after 5 warm-up screens) and median boxes.

  python scripts/eval/det_speed.py --set screenspot_pro --n 100 \
      --configs gpa:gpa:1280 ours:runs/det/axdet_1280/weights/best.pt:1280 ours_t:runs/det/axdet_1280/weights/best.pt:1280:1
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="screenspot_pro")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", default="runs/det_speed.json")
    args = ap.parse_args()
    from screen_options.detect import Detector
    screens = [json.loads(l)["image"] for l in open(DATA / "converted" / args.set / "screens.jsonl")]
    images = [Image.open(DATA / "public" / p).convert("RGB") for p in screens[:args.n + 5]]
    out_path = Path(args.out)
    results = json.load(open(out_path)) if out_path.exists() else []
    for spec in args.configs:
        name, model, sz, *tile = spec.split(":")
        det = Detector(model=model, imgsz=int(sz), tile=float(tile[0]) if tile else 0)
        ms, nb = [], []
        for im in images:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            b = det.boxes(im, (0, 0, im.width, im.height), 1.0)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
            nb.append(len(b))
        ms = sorted(ms[5:])
        r = {"set": args.set, "config": name, "imgsz": int(sz), "tile": float(tile[0]) if tile else 0,
             "ms_median": round(statistics.median(ms), 1), "ms_p90": round(ms[int(0.9 * len(ms))], 1),
             "boxes_median": statistics.median(nb[5:]), "screens": len(ms)}
        print(json.dumps(r), flush=True)
        results.append(r)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
