"""Instruction-aware signals for every (example, candidate): SigLIP image-text similarity and OCR text match.

Crops (box with a little margin, 224px) are embedded once per screen; each instruction is one text embedding.
Text match compares the instruction's words with a text candidate's OCR string (0 for icons).

  python scripts/eval/inst_feats.py --sets osworld_g screenspot_pro ... --name axgpa_t
  -> data/cands/<set>/<name>.inst.jsonl  {"image", "examples": {example id: {"sim": [...], "txt": [...]}}, "ms"}
"""
import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from rank import CANDS, load  # noqa: E402

DATA = ROOT / "data"
MODEL = "google/siglip-base-patch16-224"
STOP = set("the a an to of in on for and or with at by from is it this that click select open go button icon tab "
           "menu please then into".split())


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def text_match(inst, text):
    """Best of: share of the element's words found in the instruction, and fuzzy similarity of the strings."""
    if not text:
        return 0.0
    iw, tw = set(words(inst)), words(text)
    overlap = sum(w in iw for w in tw) / len(tw) if tw else 0.0
    fuzzy = difflib.SequenceMatcher(None, inst.lower(), text.lower()).find_longest_match().size / max(len(text), 1)
    return max(overlap, min(fuzzy, 1.0))


def vec(out):
    """transformers 5 returns a model output from get_*_features; older versions a tensor."""
    return out if torch.is_tensor(out) else out.pooler_output


def crop(img, box, style="tight", margin=0.15):
    """tight: the box plus a small margin. context: a square window about 3x the element (at least 96 px), with the
    element outlined in red, so the model sees neighbouring labels and which item is meant."""
    if style == "context":
        return context_crop(img, box)
    W, H = img.size
    x, y, w, h = box
    m = margin * max(w, h, 8)
    b = (max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m))
    if b[2] - b[0] < 2 or b[3] - b[1] < 2:
        b = (0, 0, 2, 2)
    return img.crop(b)


def context_crop(img, box, outline=(255, 0, 0)):
    from PIL import ImageDraw
    W, H = img.size
    x, y, w, h = box
    side = max(3 * max(w, h), 96)
    cx, cy = x + w / 2, y + h / 2
    b = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
    b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
    if b[2] - b[0] < 2 or b[3] - b[1] < 2:
        return img.crop((0, 0, 2, 2))
    c = img.crop(b).copy()
    lw = max(1, round(max(c.size) / 112))           # about 2 px after resizing to 224
    ImageDraw.Draw(c).rectangle([x - b[0] - lw, y - b[1] - lw, x - b[0] + w + lw, y - b[1] + h + lw], outline=outline, width=lw)
    return c


def gpu_context_crops(img, boxes, size, outline=(255, 0, 0)):
    """context_crop for every box at once on the GPU, already resized to size x size and SigLIP-normalised.

    img: HxWx3 uint8 tensor on the GPU. Same geometry as context_crop (square ~3x window, at least 96 px, clipped
    to the screen, red outline ~1/112 of the window just outside the element); the window is resampled with
    roi_align (bilinear, 2x2 samples per output pixel) instead of PIL's bicubic resize. -> N x 3 x size x size."""
    from torchvision.ops import roi_align
    H, W = img.shape[:2]
    b = torch.tensor(boxes, dtype=torch.float32, device=img.device).reshape(-1, 4)
    x, y, w, h = b.unbind(1)
    side = torch.clamp(3 * torch.maximum(w, h), min=96)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = (cx - side / 2).clamp(0, W), (cy - side / 2).clamp(0, H)
    x1, y1 = (cx + side / 2).clamp(0, W), (cy + side / 2).clamp(0, H)
    x1, y1 = torch.maximum(x1, x0 + 2), torch.maximum(y1, y0 + 2)
    rois = torch.stack([torch.zeros_like(x0), x0, y0, x1, y1], 1)
    src = img.permute(2, 0, 1).unsqueeze(0).float()
    out = roi_align(src, rois, (size, size), spatial_scale=1.0, sampling_ratio=2, aligned=True)   # N,3,S,S
    sx, sy = size / (x1 - x0), size / (y1 - y0)
    lw = torch.clamp(torch.round(torch.maximum(x1 - x0, y1 - y0) / 112), min=1)
    ox0, ox1 = (x - x0 - lw) * sx, (x + w - x0 + lw) * sx           # outer edge of the outline, output px
    oy0, oy1 = (y - y0 - lw) * sy, (y + h - y0 + lw) * sy
    t = torch.clamp(lw * sx, min=1)
    g = torch.arange(size, device=img.device, dtype=torch.float32)
    gx, gy = g.view(1, 1, -1), g.view(1, -1, 1)
    e = lambda v: v.view(-1, 1, 1)                                   # noqa: E731
    outer = (gx >= e(ox0)) & (gx <= e(ox1)) & (gy >= e(oy0)) & (gy <= e(oy1))
    inner = (gx >= e(ox0 + t)) & (gx <= e(ox1 - t)) & (gy >= e(oy0 + t)) & (gy <= e(oy1 - t))
    ring = (outer & ~inner).unsqueeze(1)
    red = torch.tensor(outline, dtype=out.dtype, device=img.device).view(1, 3, 1, 1)
    out = torch.where(ring, red, out)
    return (out / 255 - 0.5) / 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--name", default="axgpa_t")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--model", default=MODEL, help="SigLIP checkpoint (e.g. our fine-tune, runs/siglip/...)")
    ap.add_argument("--crop", default="tight", choices=("tight", "context"))
    ap.add_argument("--only", help="file of image paths: embed just these screens (e.g. a held-out split)")
    ap.add_argument("--tag", default="", help="output suffix: <name>.inst<tag>.jsonl")
    ap.add_argument("--gpu_crops", action="store_true", help="context crops cut and resized on the GPU (gpu_context_crops)")
    args = ap.parse_args()
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16).cuda().eval()
    for s in args.sets:
        out = CANDS / s / f"{args.name}.inst{args.tag}.jsonl"
        done = {json.loads(l)["image"] for l in open(out)} if out.exists() else set()
        only = set(open(args.only).read().split()) if args.only else None
        screens = [sc for sc in load(s, args.name) if sc["image"] not in done and (only is None or sc["image"] in only)]
        print(f"{s}: {len(screens)} screens", flush=True)
        with open(out, "a") as f, torch.inference_mode():
            for i, sc in enumerate(screens):
                p = DATA / "public" / sc["image"]
                img = Image.open(p if p.exists() else DATA / sc["image"]).convert("RGB")
                t0 = time.perf_counter()
                embs = []
                if args.gpu_crops:
                    size = proc.image_processor.size["height"]
                    gimg = torch.from_numpy(np.asarray(img)).cuda()
                    crops = [e["box"] for e in sc["elements"]]
                    chunks = [gpu_context_crops(gimg, crops[j:j + args.batch], size).half()
                              for j in range(0, len(crops), args.batch)]
                else:
                    crops = [crop(img, e["box"], args.crop) for e in sc["elements"]]
                    chunks = (proc(images=crops[j:j + args.batch], return_tensors="pt")["pixel_values"].cuda().half()
                              for j in range(0, len(crops), args.batch))
                for px in chunks:
                    embs.append(torch.nn.functional.normalize(vec(model.get_image_features(pixel_values=px)), dim=-1))
                img_e = torch.cat(embs) if embs else torch.zeros(0, 768, device="cuda", dtype=torch.half)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                insts = [ex["instruction"] for ex in sc["examples"]]
                tok = proc(text=insts, padding="max_length", max_length=min(64, model.config.text_config.max_position_embeddings), truncation=True, return_tensors="pt").to("cuda")
                txt_e = torch.nn.functional.normalize(vec(model.get_text_features(**tok)), dim=-1)
                sims = (txt_e @ img_e.T).float().cpu().tolist() if len(img_e) else [[] for _ in insts]
                t2 = time.perf_counter()
                exs = {ex["id"]: {"sim": [round(v, 4) for v in sim],
                                  "txt": [round(text_match(ex["instruction"], e.get("text") or ""), 3)
                                          for e in sc["elements"]]}
                       for ex, sim in zip(sc["examples"], sims)}
                f.write(json.dumps({"image": sc["image"], "examples": exs,
                                    "ms": {"crops": round((t1 - t0) * 1000), "text": round((t2 - t1) * 1000 / max(len(insts), 1), 1)}}) + "\n")
                if i % 200 == 0:
                    print(f"{s} {i}: {len(crops)} crops {(t1 - t0) * 1000:.0f} ms, text {(t2 - t1) * 1000:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
