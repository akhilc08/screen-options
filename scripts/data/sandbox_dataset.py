"""Turn the raw EC2 Mac sandbox runs into a clean screen set with an app-level split.

  .venv/bin/python scripts/data/sandbox_dataset.py clean   # data/sandbox_aws/*/*  ->  data/sandbox_ds/screens.jsonl
  .venv/bin/python scripts/data/sandbox_dataset.py split   # -> data/sandbox_ds/splits.json (frozen, by app/site)
  .venv/bin/python scripts/data/sandbox_dataset.py sources # -> data/sandbox_ds/options.jsonl (AX + fast OCR + GPA@640)

sources: Kev's runtime option list for every screen (ids a* = AX, o* = OCR-only text, d* = detector-only
icon), plus the raw OCR lines and GPA boxes down to conf 0.05 and per-source ms. ~18 min on an M-series laptop.

clean: one row per screenshot (states that share a screenshot are the same screen). Drops blank or
near-uniform frames, then near-duplicates inside each app: same window size and colour mode, 256-bit
difference hash within HAMMING bits, and AX (role, label) sets at least JACCARD alike. Each kept screen
carries its AX elements as ground-truth targets (source "ax"), the detector boxes the collector clicked
on it (source "det", no label yet), and an ax_poor flag (fewer than AX_POOR labelled window elements).

split: whole apps/sites go to train/val/test, never single screens. Apps are shuffled with a fixed seed
inside four strata (web/native x ax-poor/ok) so AX-poor apps land in every split.
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/sandbox_aws"
OUT = ROOT / "data/sandbox_ds"
HAMMING, JACCARD, AX_POOR, MIN_STD = 12, 0.8, 8, 1.0
SPLIT = {"train": 0.7, "val": 0.1, "test": 0.2}


def dhash(img, n=16):
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def hamming(a, b):
    return int(np.unpackbits(a ^ b).sum())


def app_key(s):
    return s["target"] if s["target"].startswith("web:") else s["app"]


def condition(s, modes):
    m = re.search(r"mode=(\w+)&size=(\d+x\d+)", s.get("url") or "") or re.search(r"mode=(\w+)&size=(\d+x\d+)", modes.get(s["id"], ""))
    return (m.group(1), m.group(2)) if m else ("light", f"{s['frame'][2]}x{s['frame'][3]}")


def load_runs():
    for states in sorted(RAW.glob("*/*/states.jsonl")):
        run = states.parent
        modes, det = {}, collections.defaultdict(list)
        if (run / "transitions.jsonl").exists():
            for line in open(run / "transitions.jsonl"):
                t = json.loads(line)
                modes[t["before"]] = t.get("mode") or ""
                if t["kind"] == "pixel_click" and t.get("box"):
                    det[t["before"]].append(t["box"])
        byshot = {}
        for line in open(states):
            s = json.loads(line)
            row = byshot.setdefault(s["shot"], {**s, "det": []})
            row["det"] += [b for b in det.get(s["id"], []) if b not in row["det"]]
            row.setdefault("_modes", modes)
        yield run, byshot.values()


def clean():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    kept = collections.defaultdict(list)                  # (app, mode, size) -> [(hash, axset)]
    rows = []
    for run, states in load_runs():
        rel = run.relative_to(RAW)
        for s in states:
            stats["shots"] += 1
            path = run / s["shot"]
            try:
                img = Image.open(path)
                img.load()
            except OSError:
                stats["unreadable"] += 1
                continue
            if float(np.asarray(img.convert("L").resize((256, 160))).std()) < MIN_STD:
                stats["blank"] += 1
                continue
            mode, size = condition(s, s.pop("_modes"))
            app = app_key(s)
            axset = {(e["role"], e["label"]) for e in s["elements"] if e["label"]}
            h = dhash(img)
            near = False
            for h2, ax2 in kept[(app, mode, size)]:
                if hamming(h, h2) <= HAMMING and len(axset & ax2) >= JACCARD * max(1, len(axset | ax2)):
                    near = True
                    break
            if near:
                stats["near_duplicate"] += 1
                continue
            kept[(app, mode, size)].append((h, axset))
            labelled = [e for e in s["elements"] if e["label"]]
            sid = hashlib.sha1(f"{rel}/{s['shot']}".encode()).hexdigest()[:12]
            rows.append({
                "id": sid, "app": app, "kind": "web" if app.startswith("web:") else "native",
                "title": s.get("title"), "url": s.get("url"), "mode": mode, "size": size,
                "shot": str(path.relative_to(ROOT)), "img_size": list(img.size), "scale": s.get("scale"),
                "frame": s["frame"], "ax_poor": len(labelled) < AX_POOR, "n_ax": len(s["elements"]),
                "ms": s.get("ms"),
                "targets": [{"id": e["id"], "source": "ax", "role": e["role"], "label": e["label"],
                             "context": e.get("context", []), "enabled": e.get("enabled", True), "box": e["frame"]}
                            for e in s["elements"]] +
                           [{"id": f"m{i}", "source": "ax_menubar", "role": e["role"], "label": e["label"],
                             "context": e.get("context", []), "enabled": e.get("enabled", True), "box": e.get("frame")}
                            for i, e in enumerate(s.get("menubar", []))] +
                           [{"id": f"k{i}", "source": "det_clicked", "role": "", "label": "", "box": b}
                            for i, b in enumerate(s["det"])],
            })
            stats["kept"] += 1
    with open(OUT / "screens.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    apps = collections.Counter(r["app"] for r in rows)
    stats["apps"] = len(apps)
    stats["ax_poor"] = sum(r["ax_poor"] for r in rows)
    print(json.dumps(stats, indent=1))
    print("screens per app: min", min(apps.values()), "median", sorted(apps.values())[len(apps) // 2], "max", max(apps.values()))


def split(seed=0):
    rows = [json.loads(l) for l in open(OUT / "screens.jsonl")]
    if (OUT / "splits.json").exists():
        raise SystemExit("splits.json already exists; splits are frozen. Delete it on purpose to re-split.")
    by_app = collections.defaultdict(list)
    for r in rows:
        by_app[r["app"]].append(r)
    strata = collections.defaultdict(list)
    for app, rs in by_app.items():
        poor = sum(r["ax_poor"] for r in rs) / len(rs) >= 0.5
        strata[(rs[0]["kind"], poor)].append(app)
    rng = random.Random(seed)
    assign = {}
    for key in sorted(strata):
        apps = sorted(strata[key])
        rng.shuffle(apps)
        n = len(apps)
        cut1, cut2 = round(n * SPLIT["train"]), round(n * (SPLIT["train"] + SPLIT["val"]))
        for i, app in enumerate(apps):
            assign[app] = "train" if i < cut1 else "val" if i < cut2 else "test"
    json.dump({"seed": seed, "rule": "by app/site, stratified web/native x ax-poor", "apps": dict(sorted(assign.items()))},
              open(OUT / "splits.json", "w"), indent=1)
    c = collections.Counter()
    for r in rows:
        c[(assign[r["app"]], "screens")] += 1
        c[(assign[r["app"]], "ax_poor")] += r["ax_poor"]
    for sp in SPLIT:
        print(sp, sum(v == sp for v in assign.values()), "apps", c[(sp, "screens")], "screens", c[(sp, "ax_poor")], "ax-poor")


GPA_COREML = Path.home() / ".cache/rlca/coreml/gpa_640.mlpackage"
GPA_KEEP, GPA_USE, MAX_OCR_WORDS = 0.05, 0.15, 6


def sources():
    """Kev's option list per screen: AX + fast Apple Vision OCR (<= 6 words) + GPA@640, merged as at runtime.
    Resumable: appends to options.jsonl and skips screens already done."""
    import sys
    import time
    from types import SimpleNamespace

    import objc
    import Quartz
    from ultralytics import YOLO
    sys.path.insert(0, str(ROOT / "src"))
    from screen_options.ocr import ocr_lines
    from screen_options.sources import merge

    rows = [json.loads(l) for l in open(OUT / "screens.jsonl")]
    path = OUT / "options.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    gpa = YOLO(str(GPA_COREML), task="detect")
    t_start = time.time()
    with open(path, "a") as out:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            with objc.autorelease_pool():
                shot = ROOT / r["shot"]
                src = Quartz.CGImageSourceCreateWithURL(Quartz.CFURLCreateWithFileSystemPath(None, str(shot), 0, False), None)
                cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
                f = r["frame"]
                t0 = time.perf_counter()
                ocr = [l for l in ocr_lines(cg, f, fast=True)]
                t1 = time.perf_counter()
                img = Image.open(shot).convert("RGB")
                s = img.width / f[2]
                p = gpa.predict(img, imgsz=640, conf=GPA_KEEP, device="cpu", verbose=False)[0]
                t2 = time.perf_counter()
            det = [{"box": (round(f[0] + x1 / s), round(f[1] + y1 / s), round((x2 - x1) / s), round((y2 - y1) / s)), "conf": round(c, 3)}
                   for (x1, y1, x2, y2), c in zip(p.boxes.xyxy.tolist(), p.boxes.conf.tolist())]
            ax = [SimpleNamespace(frame=tuple(t["box"]), label=t["label"], role=t["role"], context=t.get("context", []))
                  for t in r["targets"] if t["source"] == "ax" and t["box"]]
            short = [l for l in ocr if len(l["text"].split()) <= MAX_OCR_WORDS]
            items, _ = merge(ax, short, [d for d in det if d["conf"] >= GPA_USE], f)
            opts, n = [], collections.Counter()
            for it in items:
                k = "a" if "ax" in it.sources else "o" if "ocr" in it.sources else "d"
                opts.append({"id": f"{k}{n[k]}", "sources": sorted(it.sources), "role": it.role, "label": it.label,
                             "context": it.context, "box": list(it.box), "ax_index": it.ax_index})
                n[k] += 1
            out.write(json.dumps({"id": r["id"], "options": opts, "ocr": ocr, "gpa": det,
                                  "ms": {"ocr_fast": round((t1 - t0) * 1000, 1), "gpa640": round((t2 - t1) * 1000, 1)}}) + "\n")
            if i % 250 == 0:
                out.flush()
                print(f"{i}/{len(rows)} {time.time() - t_start:.0f}s", flush=True)


BLANK_STD, MISMATCH = 3.0, 0.3


def visible():
    """Mark AX options the screenshot doesn't show. The collector captured only the main window, so sheets,
    popovers, autocomplete lists and open menus (separate windows) are in the AX list but not the image, and
    Safari's toolbar is often blank. An a* option is not visible when its box is off the image or its pixels
    are flat (grayscale std < BLANK_STD). Screens with more than MISMATCH of their AX options hidden get
    capture_mismatch=true. Rewrites options.jsonl in place."""
    screens = {json.loads(l)["id"]: json.loads(l) for l in open(OUT / "screens.jsonl")}
    path = OUT / "options.jsonl"
    rows = [json.loads(l) for l in open(path)]
    hidden_total = mism = 0
    for r in rows:
        s = screens[r["id"]]
        img = np.asarray(Image.open(ROOT / s["shot"]).convert("L"), dtype=np.float32)
        f, k = s["frame"], img.shape[1] / s["frame"][2]
        n_ax = n_hidden = 0
        for o in r["options"]:
            if o["id"][0] != "a":
                o["visible"] = True
                continue
            b = o["box"]
            x0, y0 = max(0, int((b[0] - f[0]) * k)), max(0, int((b[1] - f[1]) * k))
            x1, y1 = min(img.shape[1], int((b[0] + b[2] - f[0]) * k)), min(img.shape[0], int((b[1] + b[3] - f[1]) * k))
            crop = img[y0:y1, x0:x1]
            o["visible"] = bool(crop.size and crop.std() >= BLANK_STD)
            n_ax += 1
            n_hidden += not o["visible"]
        r["hidden_ax"] = n_hidden
        r["capture_mismatch"] = bool(n_ax and n_hidden / n_ax > MISMATCH)
        hidden_total += n_hidden
        mism += r["capture_mismatch"]
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{hidden_total} hidden AX options; {mism}/{len(rows)} screens flagged capture_mismatch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["clean", "split", "sources", "visible"])
    a = ap.parse_args()
    {"clean": clean, "split": split, "sources": sources, "visible": visible}[a.step]()
