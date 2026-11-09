"""Option-source benchmark, step 2 (any platform): run one pixel source over the manifest's screenshots.

Every box is kept down to a low confidence, with its score, so the scorer can sweep thresholds offline.
Output rows: {"key", "ms", "boxes": [[x, y, w, h, conf, class], ...]} in image pixels.

Sources (name[:imgsz]):
  omni_v2, omni_v3     OmniParser icon detectors (v2 is what the pipeline runs today)
  gpa                  Salesforce GPA-GUI-Detector (YOLO)
  macpaw               Screen2AX YOLO11l, trained on macOS apps (AGPL)
  uitag                uitag-yolo11s, trained on GroundCUA desktop screens (gated)
  uidetr               UI-DETR-1 (RF-DETR-M)
  florence_rp          Florence-2-base region proposals
  screenvlm            ScreenVLM (316M) asked to list every element

  python scripts/eval/optbench_detect.py --source omni_v2:1280 [--limit 50] [--sets explore,omniact]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data/option_bench"
YOLO = {
    "omni_v2": ("microsoft/OmniParser-v2.0", "icon_detect/model.pt"),
    "omni_v3": ("microsoft/OmniParser-v2.0", "icon_detect_v3/model.pt"),
    "gpa": ("Salesforce/GPA-GUI-Detector", "model.pt"),
    "macpaw": ("macpaw-research/yolov11l-ui-elements-detection", "ui-elements-detection.pt"),
    "uitag": ("laywens/uitag-yolo11s-ui-detect-v1", "yolo-ui.pt"),
}
CONF = 0.05


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def xyxy_rows(xyxy, conf, cls):
    return [[round(x1), round(y1), round(x2 - x1), round(y2 - y1), round(float(c), 3), str(k)]
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls)]


class YoloSource:
    def __init__(self, name, imgsz, coreml=False):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO as Y
        self.imgsz = imgsz or 1280
        weights = hf_hub_download(*YOLO[name])
        if coreml:
            # Exported once per size to ~/.cache/rlca/coreml (fp16, NMS left to ultralytics); runs on the Neural Engine.
            out = Path.home() / f".cache/rlca/coreml/{name}_{self.imgsz}.mlpackage"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                exported = Y(weights).export(format="coreml", imgsz=self.imgsz, half=True)
                Path(exported).rename(out)
            weights = str(out)
        self.model = Y(weights, task="detect")
        self.coreml = coreml
        # OmniParser's own demo uses a tight NMS; the others ship with the YOLO default.
        self.iou = 0.1 if name.startswith("omni") else 0.7

    def __call__(self, img):
        dev = "cpu" if self.coreml else device()     # Core ML picks its own compute unit
        r = self.model.predict(img, imgsz=self.imgsz, conf=CONF, iou=self.iou, device=dev, verbose=False, max_det=1000)[0]
        names = r.names
        return xyxy_rows(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), [names[int(c)] for c in r.boxes.cls.tolist()])


class OmniV3:
    """icon_detect_v3 is a bare TorchScript YOLOv9-E; decoding follows the model card (top-left letterbox,
    sigmoid class logits, left/top/right/bottom distances per grid cell, NMS at IoU 0.45)."""

    def __init__(self, imgsz):
        from huggingface_hub import hf_hub_download
        self.model = torch.jit.load(hf_hub_download("microsoft/OmniParser-v2.0", "icon_detect_v3/model.pt"),
                                    map_location=device()).eval()
        self.imgsz = imgsz or 1280

    @torch.inference_mode()
    def __call__(self, img):
        import numpy as np
        from torchvision.ops import nms
        W, H = img.size
        s = min(self.imgsz / W, self.imgsz / H)
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        canvas.paste(img.resize((round(W * s), round(H * s)), Image.BILINEAR), (0, 0))
        x = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float()[None].to(device()) / 255.0
        outs = self.model(x)
        boxes, scores = [], []
        for i, stride in enumerate((8, 16, 32)):
            cls = outs[2 * i].sigmoid()[0, 0]
            l, t, r, b = outs[2 * i + 1][0]
            g = cls.shape[-1]
            gy, gx = torch.meshgrid(torch.arange(g, device=x.device, dtype=torch.float32),
                                    torch.arange(g, device=x.device, dtype=torch.float32), indexing="ij")
            cx, cy = gx + 0.5, gy + 0.5
            boxes.append(torch.stack([(cx - l) * stride, (cy - t) * stride, (cx + r) * stride, (cy + b) * stride], -1).reshape(-1, 4))
            scores.append(cls.reshape(-1))
        boxes, scores = torch.cat(boxes), torch.cat(scores)
        keep = scores > CONF
        boxes, scores = boxes[keep], scores[keep]
        keep = nms(boxes, scores, 0.45)
        boxes, scores = (boxes[keep] / s).cpu(), scores[keep].cpu()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
        return xyxy_rows(boxes.tolist(), scores.tolist(), ["interactable"] * len(scores))


class UIDetr:
    def __init__(self, imgsz):
        from huggingface_hub import hf_hub_download
        from rfdetr.detr import RFDETRMedium
        self.model = RFDETRMedium(pretrain_weights=hf_hub_download("racineai/UI-DETR-1", "model.pth"), resolution=imgsz or 1600)

    def __call__(self, img):
        d = self.model.predict(img, threshold=CONF)
        return xyxy_rows(d.xyxy.tolist(), d.confidence.tolist(), d.class_id.tolist())


class FlorenceRP:
    def __init__(self, imgsz):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        name = "florence-community/Florence-2-base-ft"
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = Florence2ForConditionalGeneration.from_pretrained(name, dtype=torch.float16).to(device()).eval()

    @torch.inference_mode()
    def __call__(self, img):
        enc = self.proc(text="<REGION_PROPOSAL>", images=img, return_tensors="pt").to(device())
        enc["pixel_values"] = enc["pixel_values"].to(torch.float16)
        ids = self.model.generate(**enc, max_new_tokens=1024, num_beams=1, do_sample=False)
        text = self.proc.batch_decode(ids, skip_special_tokens=False)[0]
        out = self.proc.post_process_generation(text, task="<REGION_PROPOSAL>", image_size=img.size)["<REGION_PROPOSAL>"]
        return [[round(x1), round(y1), round(x2 - x1), round(y2 - y1), 1.0, "region"] for x1, y1, x2, y2 in out["bboxes"]]


SCREENTAG = re.compile(r"<([a-zA-Z][a-zA-Z0-9_]*)>\s*<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")


class ScreenVLM:
    def __init__(self, imgsz):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        name = "docling-project/ScreenVLM"
        self.proc = AutoProcessor.from_pretrained(name)
        dtype = torch.bfloat16 if device() == "cuda" else torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype).to(device()).eval()
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Generate the screen representation for this UI:"}]}]
        self.prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        self.max_new = 4096

    @torch.inference_mode()
    def __call__(self, img):
        enc = self.proc(text=self.prompt, images=[img], return_tensors="pt").to(device())
        ids = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False)
        text = self.proc.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        W, H = img.size
        rows = []
        for tag, l, t, r, b in SCREENTAG.findall(text):
            l, t, r, b = (max(0, min(int(v), 500)) for v in (l, t, r, b))
            l, r = sorted((l, r))
            t, b = sorted((t, b))
            rows.append([round(l / 500 * W), round(t / 500 * H), round((r - l) / 500 * W), round((b - t) / 500 * H), 1.0, tag])
        self.last_tokens = int(ids.shape[1] - enc["input_ids"].shape[1])
        return rows


def make(spec):
    name, _, size = spec.partition(":")
    imgsz = int(size) if size else None
    if name == "omni_v3":
        return OmniV3(imgsz)
    if name in YOLO:
        return YoloSource(name, imgsz)
    if name.endswith("_coreml") and name.removesuffix("_coreml") in YOLO:
        return YoloSource(name.removesuffix("_coreml"), imgsz, coreml=True)
    return {"uidetr": UIDetr, "florence_rp": FlorenceRP, "screenvlm": ScreenVLM}[name](imgsz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--sets", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N screens per set (for Mac timing)")
    ap.add_argument("--out", default="")
    ap.add_argument("--bench", default=str(BENCH), help="benchmark dir with manifest.jsonl (v2 images are relative to it)")
    args = ap.parse_args()
    bench = Path(args.bench)
    rows = [json.loads(l) for l in open(bench / "manifest.jsonl")]
    if args.sets:
        rows = [r for r in rows if r["set"] in args.sets.split(",")]
    if args.limit:
        per = {}
        rows = [r for r in rows if per.setdefault(r["set"], []).append(r) or len(per[r["set"]]) <= args.limit]
    src = make(args.source)
    out = Path(args.out) if args.out else bench / "sources" / f"{args.source.replace(':', '@')}.jsonl"
    path = lambda r: bench / r["image"] if (bench / r["image"]).exists() else ROOT / r["image"]  # noqa: E731
    out.parent.mkdir(parents=True, exist_ok=True)
    warm = Image.open(path(rows[0])).convert("RGB")
    src(warm)                                 # load kernels before timing
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            img = Image.open(path(r)).convert("RGB")
            sync()
            t0 = time.perf_counter()
            boxes = src(img)
            sync()
            row = {"key": r["key"], "ms": round((time.perf_counter() - t0) * 1000, 1), "boxes": boxes}
            if hasattr(src, "last_tokens"):
                row["tokens"] = src.last_tokens
            f.write(json.dumps(row) + "\n")
            if i % 100 == 0:
                print(f"{args.source} {i}/{len(rows)} {row['ms']}ms {len(boxes)} boxes", flush=True)


if __name__ == "__main__":
    sys.exit(main())
