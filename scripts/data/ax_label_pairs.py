"""Icon-label training pairs from the sandbox set: where AX works, it names what the detector boxed.

For every sandbox screen, take AX elements the GPA detector also found ("det" in sources) that carry no OCR
text ("ocr" not in sources), are visible in the capture and have a label. Those are icon-like controls, and
AX gives their true name. Each becomes one pair:

  crop   the box resized to 64x64, as the captioner sees it at serving
  wide   the box with 2x its size of margin on every side (longest side 256px), for context-aware labellers
  text   "<role>: <AX label>", e.g. "button: Go back"

Near-copies (same app, role, label and box on an 8-point grid) are kept once, and each (app, label) at most
--per_label times, so Safari's toolbar doesn't swamp the set. Splits follow data/sandbox_ds/splits.json
(by app), so the test apps are never trained on.

  python3 scripts/data/ax_label_pairs.py [--workers 8] [--per_label 20]
  -> data/ax_labels/{train,val,test}.jsonl + data/ax_labels/crops/<id>_crop.png, <id>_wide.jpg
"""
import argparse
import collections
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/sandbox_ds"
OUT = ROOT / "data/ax_labels"
MAX_LABEL = 60
MIN_SIDE = 6          # points; smaller boxes are slivers


def role_word(role):
    r = role.removeprefix("AX")
    return re.sub(r"(?<!^)(?=[A-Z])", " ", r).lower() or "item"


def select(per_label):
    """-> {screen id: [(pair id, role, label, box)]} after the filters and de-duplication."""
    apps = json.load(open(DS / "splits.json"))["apps"]
    screens = {s["id"]: s for s in map(json.loads, open(DS / "screens.jsonl"))}
    seen, per = set(), collections.Counter()
    out = collections.defaultdict(list)
    for line in open(DS / "options.jsonl"):
        o = json.loads(line)
        s = screens[o["id"]]
        for e in o["options"]:
            src = set(e["sources"])
            label = (e.get("label") or "").strip()
            b = e["box"]
            if not e["id"].startswith("a") or "det" not in src or "ocr" in src or not e.get("visible"):
                continue
            if not label or len(label) > MAX_LABEL or min(b[2], b[3]) < MIN_SIDE:
                continue
            key = (s["app"], e["role"], label, *(round(v / 8) for v in b))
            if key in seen or per[(s["app"], label)] >= per_label:
                continue
            seen.add(key)
            per[(s["app"], label)] += 1
            out[o["id"]].append((f"{o['id']}_{e['id']}", e["role"], label, b))
    return screens, apps, out


def crop_screen(job):
    s, pairs = job
    img = Image.open(ROOT / s["shot"]).convert("RGB")
    fx, fy = s["frame"][0], s["frame"][1]
    k = s["scale"]
    rows = []
    for pid, role, label, (x, y, w, h) in pairs:
        px = ((x - fx) * k, (y - fy) * k, (x - fx + w) * k, (y - fy + h) * k)
        if px[2] <= 0 or px[3] <= 0 or px[0] >= img.width or px[1] >= img.height:
            continue
        mw, mh = 2 * (px[2] - px[0]), 2 * (px[3] - px[1])
        wide = (max(0, px[0] - mw), max(0, px[1] - mh), min(img.width, px[2] + mw), min(img.height, px[3] + mh))
        img.crop(px).resize((64, 64), Image.BICUBIC).save(OUT / "crops" / f"{pid}_crop.png")
        wc = img.crop(wide)
        wc.thumbnail((256, 256), Image.BICUBIC)
        wc.save(OUT / "crops" / f"{pid}_wide.jpg", quality=90)
        rows.append({"id": pid, "app": s["app"], "role": role, "label": label,
                     "text": f"{role_word(role)}: {label}", "box_px": [round(v) for v in px]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per_label", type=int, default=20)
    args = ap.parse_args()
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    screens, apps, chosen = select(args.per_label)
    jobs = [(screens[sid], pairs) for sid, pairs in chosen.items()]
    print(f"{sum(len(p) for p in chosen.values())} pairs on {len(jobs)} screens", flush=True)
    files = {sp: open(OUT / f"{sp}.jsonl", "w") for sp in ("train", "val", "test")}
    n = collections.Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, rows in enumerate(pool.map(crop_screen, jobs, chunksize=16)):
            for r in rows:
                sp = apps.get(r["app"])
                if sp in files:
                    files[sp].write(json.dumps(r) + "\n")
                    n[sp] += 1
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)} screens", flush=True)
    for f in files.values():
        f.close()
    print(dict(n))


if __name__ == "__main__":
    main()
