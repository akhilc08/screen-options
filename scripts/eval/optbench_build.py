"""Option-source benchmark, step 1 (macOS): build the screen manifest and run the sources that only exist on a Mac.

Screens and ground truth, all boxes in image pixels:
- explore:  every unique screenshot of an explorer run. Targets are the labelled command targets on that page
            ("label", menu-bar targets left out since they have no box) and elements whose press changed the screen
            ("press"). AX boxes come from the state recorded live, with the live walk time.
- omniact / screenspot_pro / osworld_g: public screenshots with human answer boxes ("gt"). No AX.

Mac-only sources written here: ax (explore only, recorded), ocr_fast, ocr_accurate (Apple Vision).
Writes data/option_bench/manifest.jsonl and data/option_bench/sources/<source>.jsonl.

  nice -n 10 .venv/bin/python scripts/eval/optbench_build.py --run data/explore/20261025-124135
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import objc
import Quartz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from screen_options.ocr import ocr_lines  # noqa: E402

OUT = ROOT / "data/option_bench"
PUB = ROOT / "data/public"
PUBLIC_SETS = {"omniact": 0, "screenspot_pro": 300, "osworld_g": 0}   # max images per set (0 = all)


def to_px(box, frame, scale):
    x, y, w, h = box
    return [round((x - frame[0]) * scale), round((y - frame[1]) * scale), round(w * scale), round(h * scale)]


def explore_rows(run):
    states, by_id = {}, {}
    for line in open(run / "states.jsonl"):
        s = json.loads(line)
        by_id[s["id"]] = s
        states.setdefault(s["shot"], s)
    targets = defaultdict(list)
    # Labelled command targets: screen ids are <run>-<state id>; option boxes are screen points.
    screens = {}
    for line in open(ROOT / "data/labels/screens.jsonl"):
        sc = json.loads(line)
        if sc["kind"] == "explore" and sc["id"].startswith(run.name):
            screens[sc["id"]] = sc
    cmds = [json.loads(l) for l in open(ROOT / "data/labels/commands.jsonl")]
    for f in sorted((ROOT / "data/labels/parts").glob("part_*.jsonl")):
        cmds += [json.loads(l) for l in open(f)]
    seen = set()
    for c in cmds:
        if c["id"] in seen or c.get("absent") or c["screen"] not in screens:
            continue
        seen.add(c["id"])
        sc = screens[c["screen"]]
        st = by_id[c["screen"].removeprefix(run.name + "-")]
        opts = {o["id"]: o for o in sc["options"]}
        boxes = [to_px(opts[t]["box"], st["frame"], st["scale"]) for t in c.get("targets") or []
                 if t in opts and opts[t].get("box")]
        if boxes:   # any of the correct targets counts, so keep them grouped
            targets[st["shot"]].append({"kind": "label", "boxes": boxes, "id": c["id"],
                                        "src": sorted({t[0] for t in c["targets"]})})
    for line in open(run / "transitions.jsonl"):
        t = json.loads(line)
        st = by_id.get(t["before"])
        if st and t["changed"] and t["element"].get("frame"):
            targets[st["shot"]].append({"kind": "press", "boxes": [to_px(t["element"]["frame"], st["frame"], st["scale"])]})
    for shot, s in states.items():
        page = to_px(s["web_frame"], s["frame"], s["scale"]) if s.get("web_frame") else None
        yield {"key": f"explore/{shot}", "set": "explore", "site": s["target"], "image": str((run / shot).relative_to(ROOT)),
               "size": [round(s["frame"][2] * s["scale"]), round(s["frame"][3] * s["scale"])], "page": page,
               "targets": targets.get(shot, [])}, s


def public_rows(name, cap):
    by_img = defaultdict(list)
    for line in open(ROOT / f"data/converted/{name}/examples.jsonl"):
        e = json.loads(line)
        if e.get("action", "click") == "click" and e.get("gt_box"):
            by_img[e["image"]].append(e)
    imgs = sorted(by_img)
    if cap and len(imgs) > cap:
        random.seed(0)
        imgs = sorted(random.sample(imgs, cap))
    sizes = {json.loads(l)["image"]: json.loads(l)["size"] for l in open(ROOT / f"data/converted/{name}/screens.jsonl")}
    for img in imgs:
        ex = by_img[img]
        yield {"key": f"{name}/{img}", "set": name, "site": ex[0].get("platform"), "image": str((PUB / img).relative_to(ROOT)),
               "size": sizes.get(img), "page": None,
               "targets": [{"kind": "gt", "boxes": [e["gt_box"]], "id": e["id"], "ui_type": e.get("ui_type")} for e in ex]}


def cgimage(path):
    src = Quartz.CGImageSourceCreateWithURL(Quartz.CFURLCreateWithFileSystemPath(None, str(path), 0, False), None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no_ocr", action="store_true")
    args = ap.parse_args()
    run = ROOT / args.run
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    rows, ax = [], []
    for row, s in explore_rows(run):
        rows.append(row)
        boxes = [to_px(e["frame"], s["frame"], s["scale"]) + [1.0, e["role"]] for e in s["elements"] if e.get("frame")]
        ax.append({"key": row["key"], "ms": s["ms"]["ax_window"], "boxes": boxes})
    for name, cap in PUBLIC_SETS.items():
        rows += list(public_rows(name, cap))
    with open(OUT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / "sources/ax.jsonl", "w") as f:
        for r in ax:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} screens, {sum(len(r['targets']) for r in rows)} targets -> {OUT / 'manifest.jsonl'}")
    if args.no_ocr:
        return
    # OCR in image pixels: frame = the image itself.
    files = {m: open(OUT / f"sources/ocr_{m}.jsonl", "w") for m in ("fast", "accurate")}
    ocr_lines(cgimage(ROOT / rows[0]["image"]), (0, 0, 10, 10))          # load Vision before timing
    for i, r in enumerate(rows):
        with objc.autorelease_pool():
            cg = cgimage(ROOT / r["image"])
            w, h = Quartz.CGImageGetWidth(cg), Quartz.CGImageGetHeight(cg)
            for m, f in files.items():
                t0 = time.perf_counter()
                lines = ocr_lines(cg, (0, 0, w, h), fast=(m == "fast"))
                ms = round((time.perf_counter() - t0) * 1000, 1)
                f.write(json.dumps({"key": r["key"], "ms": ms, "size": [w, h],
                                    "boxes": [list(l["box"]) + [l["conf"], l["text"]] for l in lines]}) + "\n")
        if i % 100 == 0:
            print(f"ocr {i}/{len(rows)}", flush=True)
    for f in files.values():
        f.close()


if __name__ == "__main__":
    main()
