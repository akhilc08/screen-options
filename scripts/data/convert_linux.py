"""Convert extra grounding data on a Linux box (no Apple Vision): OCR with RapidOCR, no detector or captions.

Writes the same data/converted/<set>/{screens,examples}.jsonl as convert_public.py, but screens only carry OCR
text lines (candidates.py adds detector boxes itself). This is training data for the rankers only; eval sets
keep the Vision OCR from convert_public.py.

  python scripts/data/convert_linux.py --set os_atlas_windows
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/data"))
from convert_public import OUT, PUB, match  # noqa: E402


def os_atlas(plat):
    d = PUB / "os_atlas"
    for i, s in enumerate(json.load(open(d / f"desktop_domain/{plat}_splited.json"))):
        img = d / f"{plat}_images" / s["img_filename"]
        if not img.exists():
            continue
        for j, e in enumerate(s["elements"]):
            if max(e["bbox"]) > 1 or min(e["bbox"]) < 0:
                continue
            yield {"id": f"osatlas-{plat}-{i}-{j}", "image": img, "split": "train", "platform": plat,
                   "instruction": e["instruction"], "action": "click", "gt_norm": e["bbox"],
                   "ui_type": e.get("data_type")}


def groundcua():
    for e in map(json.loads, open(OUT / "groundcua" / "raw.jsonl")):
        yield {**e, "image": PUB / e["image"]}


SETS = {"os_atlas_windows": lambda: os_atlas("windows"), "groundcua": groundcua}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=SETS)
    args = ap.parse_args()
    import numpy as np
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    out = OUT / args.set
    out.mkdir(parents=True, exist_ok=True)
    screens = {}
    if (out / "screens.jsonl").exists():
        screens = {s["image"]: s for s in map(json.loads, open(out / "screens.jsonl"))}
    examples = list(SETS[args.set]())
    todo = list(dict.fromkeys(str(Path(e["image"]).relative_to(PUB)) for e in examples))
    todo = [r for r in todo if r not in screens]
    print(f"{len(examples)} examples, {len(screens)} screens done, {len(todo)} to do", flush=True)
    with open(out / "screens.jsonl", "a") as f:
        for i, rel in enumerate(todo):
            img = Image.open(PUB / rel).convert("RGB")
            res, _ = ocr(np.asarray(img))
            els = []
            for k, (quad, text, _score) in enumerate(res or []):
                xs, ys = [p[0] for p in quad], [p[1] for p in quad]
                els.append({"id": f"e{k}", "role": "text", "label": text, "sources": ["ocr"],
                            "box": [round(min(xs)), round(min(ys)), round(max(xs) - min(xs)), round(max(ys) - min(ys))]})
            s = {"image": rel, "size": [img.width, img.height], "elements": els}
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            f.flush()
            screens[rel] = s
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    n = 0
    with open(out / "examples.jsonl", "w") as f:
        for ex in examples:
            rel = str(Path(ex["image"]).relative_to(PUB))
            W, H = screens[rel]["size"]
            if "gt_norm" in ex:
                x1, y1, x2, y2 = ex.pop("gt_norm")
                ex["gt_box"] = [round(x1 * W), round(y1 * H), round((x2 - x1) * W), round((y2 - y1) * H)]
            ex["target_ids"] = match(screens[rel]["elements"], gt_box=ex.get("gt_box"),
                                     gt_point=None if ex.get("gt_box") else ex.get("gt_point"))
            f.write(json.dumps({**ex, "image": rel}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} examples", flush=True)


if __name__ == "__main__":
    main()
