"""Per-stage latency of the option-list pipeline on one GPU: detect -> merge -> crop -> embed -> score.

Per screen (a fixed random sample of each eval set, one instruction each): our tiled detector + OmniParser boxes
(as candidates.py --extra), merge with the screen's OCR lines (OCR itself runs on the Mac with Apple Vision; its
time is read from convert_public.py's per-screen "ms" and reported separately), context crops of every candidate,
SigLIP image embeddings (batched, bf16), the instruction's text embedding, and the scores + sort.
GPU stages are CUDA-synchronised; the first --warmup screens are left out.

  python scripts/eval/pipeline_speed.py --model runs/siglip/ctx384/best --n 100
  -> runs/pipeline_speed.json
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/eval"))
from inst_feats import context_crop, gpu_context_crops, vec  # noqa: E402
from screen_options.sources import merge  # noqa: E402

DATA = ROOT / "data"


def stats(xs):
    xs = sorted(xs)
    return {"median": round(statistics.median(xs), 1), "p90": round(xs[int(0.9 * (len(xs) - 1))], 1),
            "mean": round(statistics.fmean(xs), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["osworld_g", "screenspot_pro", "screenspot_v2"])
    ap.add_argument("--model", default="runs/siglip/ctx384/best")
    ap.add_argument("--weights", default="runs/det/axgpa_1280/weights/best.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gpu_crops", action="store_true", help="cut/resize/normalise crops on the GPU")
    ap.add_argument("--out", default="runs/pipeline_speed.json")
    args = ap.parse_args()
    from screen_options.detect import Detector
    from transformers import AutoModel, AutoProcessor
    det = Detector(model=args.weights, imgsz=1280, tile=1, conf=0.05)
    omni = Detector(model="omni_v2", imgsz=1280, tile=1, conf=0.05)
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()
    maxlen = min(64, model.config.text_config.max_position_embeddings)

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    out = {"gpu": torch.cuda.get_device_name(), "model": args.model}
    for s in args.sets:
        screens = {r["image"]: r for r in map(json.loads, open(DATA / "converted" / s / "screens.jsonl"))}
        cands = [c for c in map(json.loads, open(DATA / "cands" / s / "axgpa_omni.jsonl"))
                 if c["image"] in screens and c["examples"]]
        random.Random(0).shuffle(cands)
        cands = cands[:args.n + args.warmup]
        T = {k: [] for k in ("detect", "merge", "crop", "embed_img", "embed_text", "score", "total_gpu", "ocr_mac")}
        ncand = []
        for i, c in enumerate(cands):
            img = Image.open(DATA / "public" / c["image"]).convert("RGB")
            frame = (0, 0, img.width, img.height)
            sc = screens[c["image"]]
            ocr = [{"box": tuple(e["box"]), "text": e["label"]} for e in sc["elements"] if "ocr" in e["sources"]]
            inst = c["examples"][0]["instruction"]
            with torch.no_grad():
                t0 = sync()
                boxes = det.boxes(img, frame, 1.0) + [{**b, "conf": b["conf"] * 0.5} for b in omni.boxes(img, frame, 1.0)]
                t1 = sync()
                items, _ = merge([], ocr, boxes, frame)
                t2 = time.perf_counter()
                if args.gpu_crops:
                    gimg = torch.from_numpy(np.asarray(img)).cuda()
                    size = proc.image_processor.size["height"]
                    bx = [list(it.box) for it in items]
                    chunks = [gpu_context_crops(gimg, bx[k:k + args.batch], size).to(torch.bfloat16)
                              for k in range(0, len(bx), args.batch)]
                    t3 = sync()
                else:
                    crops = [context_crop(img, it.box) for it in items]
                    t3 = time.perf_counter()
                    chunks = (proc(images=crops[k:k + args.batch], return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
                              for k in range(0, len(crops), args.batch))
                embs = []
                for px in chunks:
                    embs.append(torch.nn.functional.normalize(vec(model.get_image_features(pixel_values=px)).float(), dim=-1))
                ie = torch.cat(embs) if embs else torch.zeros(0, 1, device="cuda")
                t4 = sync()
                tok = proc(text=[inst], padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt").to("cuda")
                te = torch.nn.functional.normalize(vec(model.get_text_features(**tok)).float(), dim=-1)
                t5 = sync()
                order = (te @ ie.T).squeeze(0).argsort(descending=True).tolist() if len(items) else []
                t6 = sync()
            if i < args.warmup:
                continue
            for k, v in (("detect", t1 - t0), ("merge", t2 - t1), ("crop", t3 - t2), ("embed_img", t4 - t3),
                         ("embed_text", t5 - t4), ("score", t6 - t5), ("total_gpu", t6 - t0)):
                T[k].append(v * 1000)
            if sc.get("ms", {}).get("ocr") is not None:
                T["ocr_mac"].append(sc["ms"]["ocr"])
            ncand.append(len(items))
            del order
        out[s] = {k: stats(v) for k, v in T.items() if v}
        out[s]["candidates"] = stats(ncand)
        out[s]["n"] = len(ncand)
        print(s, json.dumps(out[s]), flush=True)
    json.dump(out, open(ROOT / args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
