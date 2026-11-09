"""Run every element source on the current window of each app, time it, and draw what each found.

Sources: AX window walk (on-screen, unlabelled kept), AX menu bar, Vision OCR (accurate and
fast), OmniParser detector, and captions for detector boxes nothing else explains. The merged
list tags each item with its sources. Writes, per app, the screenshot, an overlay (AX green,
OCR blue, detector-only red) and a JSON record to data/source_eval/<stamp>/.

  .venv/bin/python scripts/eval/compare_sources.py [--apps Arc Slack ...] [--no-caption]
"""
import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from AppKit import NSWorkspace  # noqa: E402
from ApplicationServices import AXUIElementCreateApplication  # noqa: E402

from screen_options.ax import app_menus, attr, walk  # noqa: E402
from screen_options.detect import Captioner, Detector  # noqa: E402
from screen_options.ocr import ocr_lines  # noqa: E402
from screen_options.screen import capture_window  # noqa: E402
from screen_options.sources import merge  # noqa: E402

COLORS = {"ax": (40, 200, 80), "ocr": (60, 120, 255), "det": (240, 50, 50)}


def ms(t0):
    return round((time.perf_counter() - t0) * 1000, 1)


def to_px(box, frame, scale):
    x, y, w, h = box
    return [(x - frame[0]) * scale, (y - frame[1]) * scale, (x - frame[0] + w) * scale, (y - frame[1] + h) * scale]


def overlay(image, items, frame, scale):
    img = image.copy()
    d = ImageDraw.Draw(img)
    for it in items:
        color = COLORS["ax"] if "ax" in it.sources else COLORS["ocr"] if "ocr" in it.sources else COLORS["det"]
        d.rectangle(to_px(it.box, frame, scale), outline=color, width=3)
    return img


def run_app(name, pid, det, cap, out):
    t0 = time.perf_counter()
    c = capture_window(pid)
    t_cap = ms(t0)
    if not c or c["image"] is None:
        return {"app": name, "skipped": "minimized" if c and c["minimized"] else "no window"}
    frame, scale, img = c["frame"], c["scale"], c["image"]
    t0 = time.perf_counter()
    ax_win = walk(c["window"], "window", skip_context={attr(c["window"], "AXTitle")}, bounds=frame, keep_unlabeled=True)
    t_ax = ms(t0)
    t0 = time.perf_counter()
    menubar = attr(AXUIElementCreateApplication(pid), "AXMenuBar")
    ax_menu = [e for m in (app_menus(menubar) if menubar else []) for e in walk(m, "menubar")]
    t_menu = ms(t0)
    t0 = time.perf_counter()
    ocr = ocr_lines(c["cgimage"], frame)
    t_ocr = ms(t0)
    t0 = time.perf_counter()
    ocr_fast = ocr_lines(c["cgimage"], frame, fast=True)
    t_ocr_fast = ms(t0)
    t0 = time.perf_counter()
    boxes = det.boxes(img, frame, scale)
    t_det = ms(t0)
    t0 = time.perf_counter()
    items, need = merge(ax_win, ocr, boxes, frame)
    t_merge = ms(t0)
    t_capt = 0.0
    if cap and need:
        t0 = time.perf_counter()
        for i, text in zip(need, cap.captions(img, [tuple(int(v) for v in to_px(items[i].box, frame, scale)) for i in need])):
            items[i].label = text
        t_capt = ms(t0)
    stem = name.replace(" ", "_")
    img.save(out / f"{stem}.png")
    overlay(img, items, frame, scale).save(out / f"{stem}_overlay.png")
    rec = {
        "app": name, "frame": frame, "scale": scale,
        "ms": {"capture": t_cap, "ax_window": t_ax, "ax_menubar": t_menu, "ocr_accurate": t_ocr,
               "ocr_fast": t_ocr_fast, "detect": t_det, "merge": t_merge, "caption": t_capt},
        "counts": {"ax_window": len(ax_win), "ax_window_labelled": sum(1 for e in ax_win if e.label),
                   "ax_menubar": len(ax_menu), "ocr_lines": len(ocr), "ocr_fast_lines": len(ocr_fast),
                   "det_boxes": len(boxes), "merged": len(items), "captioned": len(need),
                   "by_sources": dict(Counter("+".join(sorted(it.sources)) for it in items))},
        "items": [{"label": it.label, "role": it.role, "box": it.box, "sources": sorted(it.sources)} for it in items],
    }
    (out / f"{stem}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", nargs="*")
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data/source_eval" / datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apps = {a.localizedName(): a.processIdentifier() for a in NSWorkspace.sharedWorkspace().runningApplications()
            if a.activationPolicy() == 0}
    if args.apps:
        apps = {k: v for k, v in apps.items() if k in args.apps}
    det = Detector()
    cap = None if args.no_caption else Captioner()
    recs = [run_app(name, pid, det, cap, out) for name, pid in apps.items()]
    (out / "summary.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "items"} for r in recs], indent=1))
    for r in recs:
        if "skipped" in r:
            print(f"{r['app']:10s} skipped ({r['skipped']})")
            continue
        m, n = r["ms"], r["counts"]
        print(f"{r['app']:10s} ax {n['ax_window']:3d} ({n['ax_window_labelled']} lab) {m['ax_window']:6.0f}ms | "
              f"menu {n['ax_menubar']:3d} {m['ax_menubar']:5.0f}ms | ocr {n['ocr_lines']:3d} {m['ocr_accurate']:4.0f}ms "
              f"(fast {m['ocr_fast']:3.0f}) | det {n['det_boxes']:3d} {m['detect']:4.0f}ms | merged {n['merged']:3d} "
              f"| caption {n['captioned']:3d} {m['caption']:6.0f}ms | {n['by_sources']}")
    print(out)


if __name__ == "__main__":
    main()
