"""Option-source benchmark, step 3: how many targets each source (and each combination) puts in the list, and how fast.

A target counts as covered when some option's center lands inside one of its correct boxes (clicking that option
would hit it), with a small pad. Ground truth per set:
  explore/label   labelled command targets on our Arc screens (written from the AX+OCR+detector lists, so biased to them)
  explore/press   elements the explorer pressed that changed the screen (all AX elements, so AX is 100% by construction)
  omniact, screenspot_pro, osworld_g   human answer boxes on public screenshots (no AX there)

Combinations run their parts in parallel, so a combination's time per screen is the slowest part's time.
Options per screen counts boxes after merging near-duplicates (IoU > 0.5).

  python scripts/eval/optbench_score.py [--conf 0.05,0.1,0.2,0.3] [--json out.json]
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data/option_bench"
MAX_OCR_WORDS = 6
DETECTORS = ["omni_v2", "omni_v3", "gpa", "macpaw", "uitag", "uidetr", "florence_rp", "screenvlm"]


def grid_source(manifest, n):
    """Chance baseline: n boxes tiling the screen with no look at the pixels. Coverage this gets is luck."""
    out = {}
    for r in manifest:
        W, H = r["size"] or (1920, 1080)
        cols = max(1, round((n * W / H) ** 0.5))
        rows = max(1, round(n / cols))
        cw, ch = W / cols, H / rows
        out[r["key"]] = {"ms": 0.0, "boxes": [[i * cw, j * ch, cw, ch, 1.0, "grid"] for i in range(cols) for j in range(rows)]}
    return out


def load_sources(manifest):
    out = {}
    for f in sorted((BENCH / "sources").glob("*.jsonl")):
        out[f.stem] = {r["key"]: r for r in map(json.loads, open(f))}
    for n in (50, 100, 150):
        out[f"grid{n}"] = grid_source(manifest, n)
    return out


def center(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def inside(b, pt, pad):
    return b[0] - pad <= pt[0] <= b[0] + b[2] + pad and b[1] - pad <= pt[1] <= b[1] + b[3] + pad


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u else 0.0


def dedup_count(boxes):
    """Greedy, smallest first: a box is a new option unless it overlaps an already kept one at IoU > 0.5."""
    if not boxes:
        return 0
    a = np.array(boxes, dtype=np.float32)
    a = a[np.argsort(a[:, 2] * a[:, 3])]
    x1, y1, x2, y2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    area = a[:, 2] * a[:, 3]
    ix = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
    iy = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
    inter = ix * iy
    dup = inter / np.maximum(area[:, None] + area[None] - inter, 1e-6) > 0.5
    kept = np.zeros(len(a), bool)
    for i in range(len(a)):
        kept[i] = not (dup[i, :i] & kept[:i]).any()
    return int(kept.sum())


def select(row, name, conf):
    """Boxes of one source on one screen, after its filter."""
    boxes = row["boxes"]
    if name.startswith("ocr"):
        short = name.endswith("_short")
        return [b[:4] for b in boxes if not short or len(str(b[5]).split()) <= MAX_OCR_WORDS]
    if conf < 0:                     # -K: the K most confident boxes, for equal-budget comparisons
        return [b[:4] for b in sorted(boxes, key=lambda b: -b[4])[: int(-conf)]]
    return [b[:4] for b in boxes if b[4] >= conf]


def covered(target, centers, pad):
    if not len(centers):
        return False
    cx, cy = centers[:, 0], centers[:, 1]
    return any(((cx >= t[0] - pad) & (cx <= t[0] + t[2] + pad) & (cy >= t[1] - pad) & (cy <= t[1] + t[3] + pad)).any()
               for t in target["boxes"])


def evaluate(manifest, sources, combo, conf, groups):
    """-> {group: (covered, total)}, median ms per screen, mean options per screen, screens."""
    parts = []
    for p in combo:
        base = p.replace("_short", "") if p.startswith("ocr") else p
        if base not in sources:
            return None
        parts.append((p, sources[base]))
    hits, tot, times, counts, n = defaultdict(int), defaultdict(int), [], [], 0
    for r in manifest:
        rows = [s.get(r["key"]) for _, s in parts]
        if any(x is None for x in rows):
            continue
        n += 1
        boxes = [b for (p, _), row in zip(parts, rows) for b in select(row, p, conf)]
        times.append(max(row["ms"] for row in rows))
        counts.append(dedup_count(boxes))
        pad = max(4, 0.004 * (r["size"][0] if r.get("size") else 2000))
        centers = np.array([center(b) for b in boxes], dtype=np.float32).reshape(-1, 2)
        for t in r["targets"]:
            hit = covered(t, centers, pad)
            for g in groups(r, t):
                tot[g] += 1
                hits[g] += hit
    if not n:
        return None
    return {g: (hits[g], tot[g]) for g in tot}, statistics.median(times), statistics.mean(counts), n


def groups(r, t):
    if r["set"] == "explore":
        g = [f"explore/{t['kind']}"]
        if t["kind"] == "label" and not set(t["src"]) & {"w", "m"}:
            g.append("explore/label_nonAX")
        return g
    g = [r["set"]]
    if r["set"] == "screenspot_pro" and r["site"] == "macos":
        g.append("screenspot_pro/macos")
    ut = t.get("ui_type")
    if r["set"] == "screenspot_pro" and ut in ("icon", "text"):
        g.append(f"screenspot_pro/{ut}")
    return g


COLS = ["explore/label", "explore/label_nonAX", "explore/press", "omniact", "screenspot_pro", "screenspot_pro/macos",
        "screenspot_pro/icon", "osworld_g"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="0.05,0.1,0.2,0.3")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    manifest = [json.loads(l) for l in open(BENCH / "manifest.jsonl")]
    sources = load_sources(manifest)
    dets = [s for s in sources if s.split("@")[0].removesuffix("_coreml") in DETECTORS]
    combos = [("grid50",), ("grid100",), ("grid150",)]
    combos += [("ax",), ("ocr_fast",), ("ocr_fast_short",), ("ocr_accurate",), ("ocr_accurate_short",)]
    combos += [(d,) for d in dets]
    combos += [("ax", "ocr_fast_short"), ("ax", "ocr_fast")]
    combos += [("ax", "ocr_fast_short", d) for d in dets] + [("ocr_accurate", d) for d in dets]
    combos += [("ocr_fast_short", d) for d in dets]
    results = []
    for combo in combos:
        has_det = any(p in dets for p in combo)
        for conf in ([float(c) for c in args.conf.split(",")] if has_det else [0.0]):
            ev = evaluate(manifest, sources, combo, conf, groups)
            if ev is None:
                continue
            cov, ms, cnt, n = ev
            results.append({"combo": "+".join(combo), "conf": conf, "ms_p50": round(ms, 1), "options": round(cnt, 1),
                            "screens": n, "coverage": {g: [h, t] for g, (h, t) in cov.items()}})
    hdr = ["source", "conf", "ms p50", "options/screen"] + COLS
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in results:
        cells = []
        for g in COLS:
            h, t = r["coverage"].get(g, (0, 0))
            cells.append(f"{100 * h / t:.1f}" if t else "")
        print(f"| {r['combo']} | {r['conf'] or ''} | {r['ms_p50']} | {r['options']} | " + " | ".join(cells) + " |")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
