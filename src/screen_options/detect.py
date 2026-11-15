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
    "screen2ax": ("macpaw-research/yolov11l-ui-elements-detection", "ui-elements-detection.pt", 0.7),
}
COREML_DIR = Path.home() / ".cache/rlca/coreml"                # shared with rl-computer-agent's exports
CAPTION_DIR = Path.home() / ".cache/rlca/omniparser-caption"   # written by scripts/setup/convert_omniparser_caption.py


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Detector:
    def __init__(self, model="omni_v2", imgsz=1280, conf=0.05, iou=None, coreml=False, tile=0):
        """tile: when > 0, images longer than 1.25*tile*imgsz are also detected in tiles of about tile*imgsz
        (see _tiled)."""
        """model: a WEIGHTS name or a local .pt path (e.g. our AX-trained detector).
        coreml=True runs an fp16 Core ML export on the Neural Engine (exported once per size)."""
        from ultralytics import YOLO
        if model in WEIGHTS:
            from huggingface_hub import hf_hub_download
            repo, fname, default_iou = WEIGHTS[model]
            weights = hf_hub_download(repo, fname)
        else:
            weights, default_iou = str(Path(model).expanduser()), 0.7
            model = Path(weights).parent.parent.name     # runs/<name>/weights/best.pt -> <name>, for the Core ML cache
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
        self.tile = tile

    def _predict(self, image, dev):
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=dev, verbose=False)[0]
        return r.boxes.xyxy, r.boxes.conf

    def _tiled(self, image, dev):
        """Large screens: the whole image (for big elements) plus overlapping tiles about tile*imgsz on a side
        (small icons keep their detail), merged by NMS. Boxes cut by a tile edge are dropped from that tile;
        the overlap makes sure a neighbouring tile or the full view has them whole."""
        import torchvision
        xy, cf = self._predict(image, dev)
        boxes, confs = [xy], [cf]
        W, H = image.size
        side = self.tile * self.imgsz
        nx, ny = max(1, round(W / side)), max(1, round(H / side))
        tw, th = W / nx, H / ny
        ox, oy = 0.15 * tw, 0.15 * th
        for i in range(nx):
            for j in range(ny):
                x0, y0 = int(max(0, i * tw - ox)), int(max(0, j * th - oy))
                x1, y1 = int(min(W, (i + 1) * tw + ox)), int(min(H, (j + 1) * th + oy))
                b, c = self._predict(image.crop((x0, y0, x1, y1)), dev)
                edge = 2
                keep = ((b[:, 0] > edge) | (x0 == 0)) & ((b[:, 1] > edge) | (y0 == 0)) & \
                       ((b[:, 2] < x1 - x0 - edge) | (x1 == W)) & ((b[:, 3] < y1 - y0 - edge) | (y1 == H))
                b = b[keep].clone()
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                boxes.append(b)
                confs.append(c[keep])
        b, c = torch.cat(boxes), torch.cat(confs)
        k = torchvision.ops.nms(b.float(), c.float(), self.iou)
        return b[k], c[k]

    def boxes(self, image, frame, scale):
        """-> list of {"box": (x, y, w, h) in screen points, "conf"} for a window image."""
        dev = "cpu" if self.coreml else device()     # Core ML picks its own compute unit
        big = self.tile and max(image.size) > 1.25 * self.tile * self.imgsz
        xyxy, conf = self._tiled(image, dev) if big else self._predict(image, dev)
        fx, fy = frame[0], frame[1]
        out = []
        for (x1, y1, x2, y2), c in zip(xyxy.tolist(), conf.tolist()):
            out.append({"box": (round(fx + x1 / scale), round(fy + y1 / scale), round((x2 - x1) / scale), round((y2 - y1) / scale)),
                        "pixels": (int(x1), int(y1), int(x2), int(y2)), "conf": round(c, 3)})
        return out


CROP_TOKENS = 5


class Captioner:
    def __init__(self, dtype=torch.float16, model_dir=CAPTION_DIR):
        """model_dir: OmniParser's captioner, or the AX-trained labeller (scripts/train/train_labeler.py),
        which answers "<role>: <label>"."""
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_dir)
        # Crops go in at 64x64 (no resize, as OmniParser does): DaViT's 32x downsampling leaves a 2x2 grid
        # plus one pooled token, so the prompt needs 5 image placeholders, not the 577 of a 768x768 image.
        self.processor.num_image_tokens = CROP_TOKENS - self.processor.num_additional_image_tokens
        self.model = Florence2ForConditionalGeneration.from_pretrained(model_dir, dtype=dtype).to(device()).eval()
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
