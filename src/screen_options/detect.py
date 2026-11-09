"""Pixel-only element detection: a YOLO element detector (GPA-GUI-Detector or OmniParser v2) and
OmniParser's Florence-2 captioner, fine-tuned to describe what each icon does.

Boxes come back in screen points, like AX frames and OCR lines. OmniParser settings follow its demo
(box threshold 0.05, IoU 0.1, 64x64 crops, "<CAPTION>" prompt). GPA @640 via Core ML is the chosen
detector (see docs/background/option_results.md); OmniParser v2 @1280 is kept to reproduce older numbers.
"""
from pathlib import Path

import torch

# name -> (Hugging Face repo, weights file, NMS IoU). OmniParser's demo uses a tight NMS; GPA ships with the YOLO default.
WEIGHTS = {
    "omni_v2": ("microsoft/OmniParser-v2.0", "icon_detect/model.pt", 0.1),
    "gpa": ("Salesforce/GPA-GUI-Detector", "model.pt", 0.7),
}
COREML_DIR = Path.home() / ".cache/rlca/coreml"                # shared with rl-computer-agent's exports
CAPTION_DIR = Path.home() / ".cache/rlca/omniparser-caption"   # written by scripts/setup/convert_omniparser_caption.py


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Detector:
    def __init__(self, model="omni_v2", imgsz=1280, conf=0.05, iou=None, coreml=False):
        """coreml=True runs an fp16 Core ML export on the Neural Engine (exported once per size)."""
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        repo, fname, default_iou = WEIGHTS[model]
        weights = hf_hub_download(repo, fname)
        if coreml:
            out = COREML_DIR / f"{model}_{imgsz}.mlpackage"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                Path(YOLO(weights).export(format="coreml", imgsz=imgsz, half=True)).rename(out)
            weights = str(out)
        self.model = YOLO(weights, task="detect")
        self.imgsz, self.conf = imgsz, conf
        self.iou = default_iou if iou is None else iou
        self.coreml = coreml

    def boxes(self, image, frame, scale):
        """-> list of {"box": (x, y, w, h) in screen points, "conf"} for a window image."""
        dev = "cpu" if self.coreml else device()     # Core ML picks its own compute unit
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=dev, verbose=False)[0]
        fx, fy = frame[0], frame[1]
        out = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            out.append({"box": (round(fx + x1 / scale), round(fy + y1 / scale), round((x2 - x1) / scale), round((y2 - y1) / scale)),
                        "pixels": (int(x1), int(y1), int(x2), int(y2)), "conf": round(c, 3)})
        return out


CROP_TOKENS = 5


class Captioner:
    def __init__(self, dtype=torch.float16):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(CAPTION_DIR)
        # Crops go in at 64x64 (no resize, as OmniParser does): DaViT's 32x downsampling leaves a 2x2 grid
        # plus one pooled token, so the prompt needs 5 image placeholders, not the 577 of a 768x768 image.
        self.processor.num_image_tokens = CROP_TOKENS - self.processor.num_additional_image_tokens
        self.model = Florence2ForConditionalGeneration.from_pretrained(CAPTION_DIR, dtype=dtype).to(device()).eval()
        self.dtype = dtype

    @torch.inference_mode()
    def captions(self, image, pixel_boxes, batch=64, max_new_tokens=20):
        """-> one short description per (x1, y1, x2, y2) pixel box of `image`."""
        crops = [image.crop(b).resize((64, 64)) for b in pixel_boxes]
        out = []
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            # 64x64 crops as-is, like OmniParser: the default resize to 768x768 costs ~150x the memory and time.
            enc = self.processor(text=["<CAPTION>"] * len(chunk), images=chunk, return_tensors="pt",
                                 do_resize=False).to(device())
            enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
            ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
            out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out
