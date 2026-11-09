"""Convert public grounding datasets into Kev's element-list format.

None of these sets ship the element list Kev picks from, so each screenshot gets one from the same pixel
sources we run on the Mac (Apple Vision OCR + OmniParser detector, icons captioned). The dataset's answer
box (or click point) is then matched to the elements it covers. AX is not available here, so these lists
are noisier than real Mac ones.

Writes data/converted/<set>/screens.jsonl (one row per screenshot: its elements) and
data/converted/<set>/examples.jsonl (one row per instruction: target element ids, action, split).
Examples whose target no element covers are kept with "target_ids": [], which measures how often the
pixel sources miss the target. Resumable: screenshots already in screens.jsonl are skipped.

ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G and MMBench-GUI (level 2) are public benchmarks, so they get split
"eval" and stay out of training. ScreenSpot v1 is skipped (v2 is its corrected re-release).

  .venv/bin/python scripts/data/convert_public.py --set screenspot_v2 [--limit 20]
  sets: screenspot_v2 screenspot_pro omniact os_atlas_macos os_atlas_linux osworld_g mmbench_gui
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "data/public"
OUT = ROOT / "data/converted"


def xywh(x1, y1, x2, y2):
    return [x1, y1, x2 - x1, y2 - y1]


def screenspot_v2():
    d = PUB / "screenspot_v2"
    for kind in ("desktop", "web", "mobile"):
        for i, r in enumerate(json.load(open(d / f"screenspot_{kind}_v2.json"))):
            yield {"id": f"ssv2-{kind}-{i}", "image": d / "screenspotv2_image" / r["img_filename"], "split": "eval",
                   "platform": r.get("data_source"), "instruction": r["instruction"], "action": "click",
                   "gt_box": r["bbox"], "ui_type": r.get("data_type")}


def screenspot_pro():
    d = PUB / "screenspot_pro"
    for f in sorted(glob.glob(str(d / "annotations/*.json"))):
        for r in json.load(open(f)):
            yield {"id": f"sspro-{r['id']}", "image": d / "images" / r["img_filename"], "split": "eval",
                   "platform": r.get("platform"), "app": r.get("application"), "instruction": r["instruction"],
                   "action": "click", "gt_box": xywh(*r["bbox"]), "ui_type": r.get("ui_type")}


def osworld_g():
    d = PUB / "osworld_g"     # annotations.json + images/ extracted from the HF parquet
    for r in json.load(open(d / "annotations.json")):
        c = r["box_coordinates"]
        if r["box_type"] == "polygon":      # x, y pairs: their bounding rectangle
            xs, ys = c[0::2], c[1::2]
            c = xywh(min(xs), min(ys), max(xs), max(ys))
        yield {"id": f"osworldg-{r['id']}", "image": d / "images" / r["image_path"], "split": "eval", "platform": "linux",
               "instruction": r["instruction"], "action": "click", "gt_box": [round(v) for v in c], "ui_type": r["GUI_types"]}


def mmbench_gui():
    """Level 2 (element grounding) only; boxes are normalised x1, y1, x2, y2."""
    d = PUB / "mmbench_gui"
    for r in json.load(open(d / "L2_annotations.json")):
        yield {"id": f"mmbench-{r['index']}", "image": d / "offline_images" / r["platform"] / r["image_path"], "split": "eval",
               "platform": r["platform"].removeprefix("os_"), "app": r.get("app_name"), "instruction": r["instruction"],
               "action": "click", "gt_norm": r["bbox"], "ui_type": r.get("data_type"), "grounding": r.get("grounding_type")}


ACT = re.compile(r"pyautogui\.(click|doubleClick|rightClick|moveTo|write|press|hotkey)\((.*)\)")
ACTION_NAMES = {"click": "click", "doubleClick": "double_click", "rightClick": "right_click", "moveTo": "hover"}


def omniact_step(script):
    """-> (action, (x, y), text) for the first on-screen step of a pyautogui script, or None if it has none.
    A click followed by write() becomes one "type" step: Kev picks the field and says what text goes in."""
    steps = [m.groups() for m in map(ACT.search, script.splitlines()) if m]
    for i, (fn, args) in enumerate(steps):
        if fn not in ACTION_NAMES:
            continue
        nums = re.findall(r"-?\d+(?:\.\d+)?", args)
        if len(nums) < 2:
            return None
        pt = (float(nums[0]), float(nums[1]))
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        if fn == "click" and nxt and nxt[0] == "write":
            return "type", pt, nxt[1].strip().strip("'\"")
        return ACTION_NAMES[fn], pt, None
    return None


def omniact_box(d, image, pt):
    """-> smallest annotated UI box (x, y, w, h) containing the click point, or None.
    Scripts only give a point; the boxes file next to each screen says which element that point is on."""
    p = Path(image)
    folder = d / "data/metadata" / p.parts[2] / "boxes" / p.parts[3]
    # desktop screens use screen_1.json, web screens screen1_boxes.json
    f = next((f for f in (folder / f"{p.stem}.json", folder / f"{p.stem.replace('_', '')}_boxes.json") if f.exists()), None)
    if not f:
        return None
    boxes = [xywh(*b["top_left"], *b["bottom_right"]) for b in json.load(open(f)).values() if b.get("valid", 1)]
    hits = [b for b in boxes if inside(b, pt)]
    return min(hits, key=lambda b: b[2] * b[3]) if hits else None


def omniact():
    d = PUB / "omniact"
    for split in ("train", "val", "test"):
        for k, r in json.load(open(d / f"{split}.json")).items():
            text = (d / r["task"]).read_text()
            head, script = (re.split(r"Output script:", text, maxsplit=1, flags=re.I) + [""])[:2]
            task = head.removeprefix("Task:").strip()
            step = omniact_step(script)
            if not step:
                continue
            action, pt, typed = step
            box = omniact_box(d, r["image"], pt)
            yield {"id": f"omniact-{split}-{k}", "image": d / r["image"], "split": split,
                   "platform": r["image"].split("/")[2], "app": r["image"].split("/")[3], "instruction": task,
                   "action": action, "text": typed, **({"gt_box": box} if box else {}), "gt_point": pt,
                   "multi_step": text.count("pyautogui.") > (2 if typed else 1)}


def os_atlas(plat):
    d = PUB / "os_atlas"
    for i, s in enumerate(json.load(open(d / f"desktop_domain/{plat}_splited.json"))):
        img = d / f"{plat}_images" / s["img_filename"]
        for j, e in enumerate(s["elements"]):
            if max(e["bbox"]) > 1 or min(e["bbox"]) < 0:      # boxes past the screen edge are annotation noise
                continue
            yield {"id": f"osatlas-{plat}-{i}-{j}", "image": img, "split": "train", "platform": plat,
                   "instruction": e["instruction"], "action": "click", "gt_norm": e["bbox"], "ui_type": e["data_type"]}


SETS = {"screenspot_v2": screenspot_v2, "screenspot_pro": screenspot_pro, "omniact": omniact,
        "os_atlas_macos": lambda: os_atlas("macos"), "os_atlas_linux": lambda: os_atlas("linux"),
        "osworld_g": osworld_g, "mmbench_gui": mmbench_gui}


class Elements:
    """Pixel-source element lists for whole screenshots, boxes in image pixels."""

    def __init__(self):
        sys.path.insert(0, str(ROOT / "src"))
        from screen_options.detect import Captioner, Detector
        self.det, self.cap = Detector(), Captioner()

    def __call__(self, path):
        import Quartz
        from PIL import Image
        from screen_options.ocr import ocr_lines
        from screen_options.sources import merge
        img = Image.open(path).convert("RGB")
        frame = (0, 0, img.width, img.height)
        src = Quartz.CGImageSourceCreateWithURL(Quartz.CFURLCreateWithFileSystemPath(None, str(path), 0, False), None)
        t0 = time.perf_counter()
        ocr = ocr_lines(Quartz.CGImageSourceCreateImageAtIndex(src, 0, None), frame)
        t1 = time.perf_counter()
        boxes = self.det.boxes(img, frame, 1.0)
        t2 = time.perf_counter()
        items, need = merge([], ocr, boxes, frame)
        px = [(items[j].box[0], items[j].box[1], items[j].box[0] + items[j].box[2], items[j].box[1] + items[j].box[3])
              for j in need]
        for j, text in zip(need, self.cap.captions(img, px) if px else []):
            items[j].label = text
        t3 = time.perf_counter()
        elements = [{"id": f"e{i}", "role": it.role, "label": it.label, "box": list(it.box), "sources": sorted(it.sources)}
                    for i, it in enumerate(items)]
        ms = {"ocr": round((t1 - t0) * 1000), "detect": round((t2 - t1) * 1000), "caption": round((t3 - t2) * 1000)}
        return {"size": [img.width, img.height], "elements": elements, "ms": ms}


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def inside(box, pt):
    return box[0] <= pt[0] <= box[0] + box[2] and box[1] <= pt[1] <= box[1] + box[3]


def match(elements, gt_box=None, gt_point=None, max_area_ratio=4.0):
    """-> element ids that hit the answer, best first.

    Box answers: elements whose centre is in the box or that overlap it (IoU > 0.3), best by IoU.
    Point answers: elements containing the point, smallest first. Elements far bigger than the answer box
    (containers, whole paragraphs) don't count, so a hit means Kev could really pick the right thing."""
    hits = []
    for e in elements:
        b = e["box"]
        if gt_box is not None:
            c = (b[0] + b[2] / 2, b[1] + b[3] / 2)
            score = iou(b, gt_box)
            if (inside(gt_box, c) or score > 0.3) and b[2] * b[3] <= max_area_ratio * max(gt_box[2] * gt_box[3], 1):
                hits.append((-score, e["id"]))
        elif inside(b, gt_point):
            hits.append((b[2] * b[3], e["id"]))
    return [i for _, i in sorted(hits)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many screenshots (for a quick check)")
    ap.add_argument("--image_timeout", type=float, default=120, help="seconds before a hung screenshot ends the run")
    args = ap.parse_args()
    import objc
    import torch
    out = OUT / args.set
    out.mkdir(parents=True, exist_ok=True)
    screens = {}
    if (out / "screens.jsonl").exists():
        for line in open(out / "screens.jsonl"):
            s = json.loads(line)
            screens[s["image"]] = s
    examples = [ex for ex in SETS[args.set]() if Path(ex["image"]).exists()]
    todo = list(dict.fromkeys(str(Path(ex["image"]).relative_to(PUB)) for ex in examples if str(Path(ex["image"]).relative_to(PUB)) not in screens))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(examples)} examples, {len(screens)} screenshots done, {len(todo)} to do", flush=True)
    if todo:
        get = Elements()
        pending = out / "pending.txt"     # the image in progress; still here after a hang means that image hung
        with open(out / "screens.jsonl", "a") as f:
            for i, rel in enumerate(todo):
                if pending.exists() and pending.read_text() == rel:
                    s = {"image": rel, "size": None, "elements": [], "error": "hung on a previous run"}
                else:
                    pending.write_text(rel)
                    # Vision OCR has deadlocked on one 4K screenshot; exit so a rerun can skip it.
                    watchdog = threading.Timer(args.image_timeout, lambda: os._exit(3))
                    watchdog.start()
                    # Vision objects are autoreleased: without a pool per image they pile up (29 GB once in the replay).
                    with objc.autorelease_pool():
                        s = {"image": rel, **get(PUB / rel)}
                    watchdog.cancel()
                pending.unlink(missing_ok=True)
                f.write(json.dumps(s) + "\n")
                f.flush()
                screens[rel] = s
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if (i + 1) % 25 == 0:
                    print(f"{i + 1}/{len(todo)}", flush=True)
    n = hit = 0
    with open(out / "examples.jsonl", "w") as f:
        for ex in examples:
            rel = str(Path(ex["image"]).relative_to(PUB))
            if rel not in screens:
                continue
            s = screens[rel]
            if not s["size"]:
                continue
            W, H = s["size"]
            if "gt_norm" in ex:
                x1, y1, x2, y2 = ex.pop("gt_norm")
                ex["gt_box"] = [round(x1 * W), round(y1 * H), round((x2 - x1) * W), round((y2 - y1) * H)]
            ids = match(s["elements"], gt_box=ex.get("gt_box"), gt_point=None if ex.get("gt_box") else ex.get("gt_point"))
            f.write(json.dumps({**ex, "image": rel, "target_ids": ids}) + "\n")
            n += 1
            hit += bool(ids)
    print(f"wrote {n} examples; target found in the element list for {hit} ({hit / max(n, 1):.0%})")


if __name__ == "__main__":
    main()
