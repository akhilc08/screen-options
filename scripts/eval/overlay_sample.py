"""By-eye check of what OCR and the detector add on top of AX, on a sample of replayed screenshots.

draw:  for a few screenshots per site, crop the page area and draw AX items in green, OCR-only items in
       blue and detector-only items in red, each extra numbered. Writes <run>/overlays/<shot>.png and a
       <run>/overlays/judgments.jsonl template with one row per screenshot.
score: after "clickable" (numbers of the extras that are real targets) and "missed" (targets nothing
       found) are filled in by hand, prints how many of the extras are real, per site and overall.
       A crowded screen can instead get "real_counts": {"ocr": n, "det": n} with "clickable": [].

  .venv/bin/python scripts/eval/overlay_sample.py draw --run data/explore/<stamp> --per_site 3
  .venv/bin/python scripts/eval/overlay_sample.py score --run data/explore/<stamp>
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {"ax": (0, 170, 0), "ocr": (0, 90, 255), "det": (230, 0, 0)}


def center(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def inside(frame, pt):
    return frame[0] <= pt[0] <= frame[0] + frame[2] and frame[1] <= pt[1] <= frame[1] + frame[3]


def kind(item):
    return "ax" if "ax" in item["sources"] else "ocr" if "ocr" in item["sources"] else "det"


def page_items(row, state):
    """Items whose centre is in the web area, as (kind, item); the browser sidebar and toolbar are left out."""
    web = state.get("web_frame") or state["frame"]
    return [(kind(it), it) for it in row["items"] if inside(web, center(it["box"]))]


def draw(run, row, state, out, zoom=1.5):
    frame, scale = state["frame"], state["scale"]
    web = state.get("web_frame") or frame
    img = Image.open(run / row["shot"]).convert("RGB")
    px = lambda x, y: ((x - frame[0]) * scale, (y - frame[1]) * scale)
    crop = (*px(web[0], web[1]), *px(web[0] + web[2], web[1] + web[3]))
    img = img.crop(tuple(int(v) for v in crop))
    k = zoom / scale
    img = img.resize((int(img.width * k), int(img.height * k)))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=13)
    extras = []
    for src, it in page_items(row, state):
        x, y, w, h = it["box"]
        box = [(x - web[0]) * zoom, (y - web[1]) * zoom, (x - web[0] + w) * zoom, (y - web[1] + h) * zoom]
        d.rectangle(box, outline=COLORS[src], width=1 if src == "ax" else 2)
        if src != "ax":
            n = len(extras)
            extras.append({"n": n, "kind": src, "label": it["label"], "box": it["box"]})
            tx, ty = box[0], max(0, box[1] - 14)
            d.rectangle([tx, ty, tx + 8 * len(str(n)) + 4, ty + 14], fill=COLORS[src])
            d.text((tx + 2, ty), str(n), fill="white", font=font)
    img.save(out)
    return extras


def cmd_draw(args):
    run = Path(args.run)
    states = {}
    for line in open(run / "states.jsonl"):
        s = json.loads(line)
        states.setdefault(s["shot"], s)
    by_site = defaultdict(list)
    for line in open(run / "sources.jsonl"):
        r = json.loads(line)
        by_site[r["target"]].append(r)
    random.seed(args.seed)
    odir = run / "overlays"
    odir.mkdir(exist_ok=True)
    with open(odir / "judgments.jsonl", "w") as f:
        for site, rows in sorted(by_site.items()):
            for r in random.sample(rows, min(args.per_site, len(rows))):
                s = states[r["shot"]]
                out = odir / (Path(r["shot"]).stem + ".png")
                extras = draw(run, r, s, out)
                n_ax = sum(1 for src, _ in page_items(r, s) if src == "ax")
                f.write(json.dumps({"shot": r["shot"], "site": site, "url": s.get("url"), "overlay": str(out.relative_to(run)),
                                    "ax": n_ax, "extras": extras, "clickable": None, "missed": None}) + "\n")
                print(out)


def cmd_score(args):
    rows = [json.loads(l) for l in open(Path(args.run) / "overlays" / "judgments.jsonl")]
    rows = [r for r in rows if r["clickable"] is not None]
    groups = defaultdict(list)
    for r in rows:
        groups[r["site"]].append(r)
    groups["ALL"] = rows
    print(f"{'site':10} {'screens':>7} {'ax':>5} {'ocr+':>5} {'real':>5} {'det+':>5} {'real':>5} {'missed':>6}")
    for site, rs in groups.items():
        n = len(rs)
        tally = defaultdict(int)
        for r in rs:
            real = set(r["clickable"])
            for e in r["extras"]:
                tally[e["kind"]] += 1
                tally[e["kind"] + "_real"] += e["n"] in real
            for k, v in (r.get("real_counts") or {}).items():   # for crowded screens judged by count, not by number
                tally[k + "_real"] += v
            tally["ax"] += r["ax"]
            tally["missed"] += r["missed"] or 0
        f = lambda k: tally[k] / n
        print(f"{site:10} {n:7} {f('ax'):5.1f} {f('ocr'):5.1f} {f('ocr_real'):5.1f} {f('det'):5.1f} {f('det_real'):5.1f} {f('missed'):6.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--run", required=True)
    d.add_argument("--per_site", type=int, default=3)
    d.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    args = ap.parse_args()
    cmd_draw(args) if args.cmd == "draw" else cmd_score(args)


if __name__ == "__main__":
    main()
