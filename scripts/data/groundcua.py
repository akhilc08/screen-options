"""Sample GroundCUA (ServiceNow, MIT) screens + instructions as ranker training data.

GroundCUA: dense desktop screenshots from 87 open-source apps with human-verified boxes (~64 elements/screen,
small targets), close to ScreenSpot-Pro's density. Its instruction_tuning release has, per element, functional /
direct / icon instructions (boxes) and spatial ones ("the element left of X", points). We sample --n screens,
download their images and write raw examples for convert_linux.py (OCR) and candidates.py.

  python scripts/data/groundcua.py --n 4000
  -> data/public/groundcua/images/<platform>/<hash>.png, data/converted/groundcua/raw.jsonl
"""
import argparse
import json
import random
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/public/groundcua"
BASE = "https://huggingface.co/datasets/ServiceNow/GroundCUA/resolve/main/images/"
FILES = ["functional_instructions.json", "direct_icon_instructions.json", "direct_general_templates_instructions.json",
         "direct_text_instructions.json", "direct_miscellaneous_instructions.json", "remaining_instructions.json"]
KEYS = ("functional_instruction", "logo_and_icon_instruction", "general_instruction", "text_instruction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--per_screen", type=int, default=24, help="max examples kept per screen")
    args = ap.parse_args()
    rng = random.Random(0)
    src = D / "groundcua_data"
    first = json.load(open(src / FILES[0]))
    ids = sorted({(e["platform"], e["id"]) for e in first})
    rng.shuffle(ids)
    keep = dict((i, p) for p, i in ids[:args.n])
    del first
    ex = {i: [] for i in keep}
    for f in FILES:
        for e in json.load(open(src / f)):
            if e["id"] not in keep or not e.get("bbox"):
                continue
            ins = e.get("all_instructions") or {}
            opts = [ins[k] for k in KEYS if ins.get(k)]
            if e.get("text") and e.get("category") in ("Button", "Visual Elements", "Menu", "Navigation"):
                opts.append(e["text"])          # terse label-style commands, like ScreenSpot-Pro's
            if opts:
                x1, y1, x2, y2 = e["bbox"]
                ex[e["id"]].append({"instruction": rng.choice(opts),
                                    "gt_box": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)]})
        print(f"read {f}", flush=True)
    for e in json.load(open(src / "spatial_data.json")):
        if e["id"] in keep:
            ex[e["id"]].append({"instruction": e["instruction"], "gt_point": [round(c) for c in e["cordinate"]]})

    def fetch(i):
        p = D / "images" / keep[i] / f"{i}.png"
        if not p.exists():
            r = requests.get(BASE + urllib.parse.quote(f"{keep[i]}/{i}.png"), timeout=120)
            if r.status_code != 200:
                return 0
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(r.content)
        return 1

    with ThreadPoolExecutor(8) as pool:
        ok = dict(zip(keep, pool.map(fetch, keep)))
    out = ROOT / "data/converted/groundcua"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "raw.jsonl", "w") as f:
        for i, p in keep.items():
            if not ok[i]:
                continue
            exs = ex[i] if len(ex[i]) <= args.per_screen else rng.sample(ex[i], args.per_screen)
            for k, e in enumerate(exs):
                f.write(json.dumps({"id": f"gcua-{i[:12]}-{k}", "image": f"groundcua/images/{p}/{i}.png",
                                    "split": "train", "platform": p, "action": "click", **e}) + "\n")
                n += 1
    print(f"{sum(ok.values())} images, {n} examples", flush=True)


if __name__ == "__main__":
    main()
