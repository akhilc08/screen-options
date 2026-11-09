"""Option benchmark v2: score every option source, combination and merge rule on dense ground truth, per app.

For each screen, a combination yields an option list. Metrics:
  recall (clickable)  share of clickable targets with an option center inside them (clicking the option hits it)
  recall (all)        same over every annotated element, clickable or not
  precision           share of options whose center lands on any annotated element (the rest is noise in Kev's list)
  options/screen      after merging near-duplicates (IoU > 0.5)
  ms                  parts run in parallel, so the slowest part's time

Merge rules for lists that mix structured (ax, ocr) and pixel (detector) sources:
  union    everything, near-duplicates merged
  fill     structured sources first; a detector box is added only if it overlaps no kept box (IoU <= 0.1), as UFO2 does
  gateK    pixel sources only on screens where AX has fewer than K elements (sandbox set, where AX exists)

  python scripts/eval/optbench2_score.py --bench data/option_bench2 [--conf 0.05,0.15,0.3] [--json out.json]
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

MAX_OCR_WORDS = 6
STRUCTURED = ("ax", "ocr_fast", "ocr_accurate")


def load(bench):
    manifest = [json.loads(l) for l in open(bench / "manifest.jsonl")]
    sources = {}
    for f in sorted((bench / "sources").glob("*.jsonl")):
        sources[f.stem] = {r["key"]: r for r in map(json.loads, open(f))}
    return manifest, sources


def pick(row, part, conf):
    """Boxes of one part on one screen. ocr_*_short keeps lines of at most 6 words (labels, not paragraphs)."""
    if part.startswith("ocr"):
        short = part.endswith("_short")
        return [b[:4] for b in row["boxes"] if not short or len(str(b[5]).split()) <= MAX_OCR_WORDS]
    if part == "ax":
        return [b[:4] for b in row["boxes"]]
    return [b[:4] for b in row["boxes"] if b[4] >= conf]


def iou_matrix(a, b):
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    iy = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = ix * iy
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-6)


def dedup(boxes):
    """Greedy, smallest first: keep a box unless it overlaps an already kept one at IoU > 0.5."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][2] * boxes[i][3])
    a = [boxes[i] for i in order]
    dup = iou_matrix(a, a) > 0.5
    kept = np.zeros(len(a), bool)
    for i in range(len(a)):
        kept[i] = not (dup[i, :i] & kept[:i]).any()
    return [b for b, k in zip(a, kept) if k]


def build_list(r, parts, sources, conf, rule):
    """-> (options, ms) or None when a part has no row for this screen."""
    rows = []
    for p in parts:
        base = p.removesuffix("_short")
        row = sources.get(base, {}).get(r["key"])
        if row is None:
            return None
        rows.append((p, row))
    ms = max(row["ms"] for _, row in rows)
    struct = [b for p, row in rows if p.removesuffix("_short") in STRUCTURED for b in pick(row, p, conf)]
    pixel = [b for p, row in rows if p.removesuffix("_short") not in STRUCTURED for b in pick(row, p, conf)]
    if rule.startswith("gate"):
        ax_row = sources.get("ax", {}).get(r["key"])
        if ax_row is None:
            return None
        if len(ax_row["boxes"]) >= int(rule[4:]):
            pixel = []
            ms = max(row["ms"] for p, row in rows if p.removesuffix("_short") in STRUCTURED)
    if rule == "fill" and struct and pixel:
        keep = iou_matrix(pixel, struct).max(axis=1) <= 0.1
        pixel = [b for b, k in zip(pixel, keep) if k]
    return dedup(struct + pixel), ms


def hits(points, boxes, pad):
    """For each box, whether any point falls inside it (with pad). points (N,2), boxes (M,4) -> bool (M,)."""
    if not len(points) or not len(boxes):
        return np.zeros(len(boxes), bool)
    p, b = np.asarray(points, np.float32), np.asarray(boxes, np.float32)
    inx = (p[:, 0][None] >= b[:, 0][:, None] - pad) & (p[:, 0][None] <= (b[:, 0] + b[:, 2])[:, None] + pad)
    iny = (p[:, 1][None] >= b[:, 1][:, None] - pad) & (p[:, 1][None] <= (b[:, 1] + b[:, 3])[:, None] + pad)
    return (inx & iny).any(axis=1)


def score(manifest, sources, parts, conf, rule, key_fn):
    agg = defaultdict(lambda: {"click": [0, 0], "all": [0, 0], "prec": [0, 0], "opts": [], "ms": [], "screens": 0})
    for r in manifest:
        built = build_list(r, parts, sources, conf, rule)
        if built is None:
            continue
        opts, ms = built
        pad = max(3, 0.003 * r["size"][0])
        centers = np.array([[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in opts], np.float32).reshape(-1, 2)
        tboxes = [t["boxes"][0] for t in r["targets"]]
        hit = hits(centers, tboxes, pad)
        click = np.array([t["clickable"] for t in r["targets"]], bool)
        # Precision: an option is useful when its center lies inside some annotated element.
        useful = 0
        if len(opts) and tboxes:
            t, c = np.asarray(tboxes, np.float32), centers
            inside = ((c[:, 0][:, None] >= t[:, 0][None] - pad) & (c[:, 0][:, None] <= (t[:, 0] + t[:, 2])[None] + pad) &
                      (c[:, 1][:, None] >= t[:, 1][None] - pad) & (c[:, 1][:, None] <= (t[:, 1] + t[:, 3])[None] + pad))
            useful = int(inside.any(axis=1).sum())
        for g in key_fn(r):
            a = agg[g]
            a["click"][0] += int(hit[click].sum())
            a["click"][1] += int(click.sum())
            a["all"][0] += int(hit.sum())
            a["all"][1] += len(hit)
            a["prec"][0] += useful
            a["prec"][1] += len(opts)
            a["opts"].append(len(opts))
            a["ms"].append(ms)
            a["screens"] += 1
    out = {}
    for g, a in agg.items():
        pct = lambda h_t: round(100 * h_t[0] / h_t[1], 1) if h_t[1] else None  # noqa: E731
        out[g] = {"recall_click": pct(a["click"]), "recall_all": pct(a["all"]), "precision": pct(a["prec"]),
                  "options": round(statistics.mean(a["opts"]), 1), "ms_p50": round(statistics.median(a["ms"]), 1),
                  "screens": a["screens"]}
    return out


def combos(sources):
    dets = [s for s in sources if s not in STRUCTURED]
    has_ax = "ax" in sources
    out = [((s,), "union") for s in ("ax", "ocr_fast", "ocr_fast_short", "ocr_accurate", "ocr_accurate_short") if s.removesuffix("_short") in sources]
    out += [((d,), "union") for d in dets]
    for d in dets:
        out += [(("ocr_fast_short", d), "union"), (("ocr_fast_short", d), "fill"), (("ocr_accurate_short", d), "fill")]
        if has_ax:
            out += [(("ax", "ocr_fast_short", d), rule) for rule in ("union", "fill", "gate10", "gate25")]
    if has_ax:
        out += [(("ax", "ocr_fast_short"), "union"), (("ax", "ocr_fast"), "union")]
    for i, a in enumerate(dets):                      # two detectors together
        for b in dets[i + 1:]:
            if a.split("@")[0] != b.split("@")[0]:
                out.append((("ocr_fast_short", a, b), "fill"))
    return [(p, r) for p, r in out if all(x.removesuffix("_short") in sources for x in p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/option_bench2")
    ap.add_argument("--conf", default="0.05,0.15,0.3")
    ap.add_argument("--json", default="")
    ap.add_argument("--per_app_top", type=int, default=6, help="per-app table for the N best combos")
    args = ap.parse_args()
    bench = Path(args.bench)
    manifest, sources = load(bench)
    results = []
    for parts, rule in combos(sources):
        has_det = any(p.removesuffix("_short") not in STRUCTURED for p in parts)
        for conf in ([float(c) for c in args.conf.split(",")] if has_det else [0.0]):
            res = score(manifest, sources, parts, conf, rule, lambda r: [r["set"]])
            if res:
                results.append({"combo": "+".join(parts), "rule": rule, "conf": conf, "by_set": res})
    sets = sorted({r["set"] for r in manifest})
    for s in sets:
        rows = sorted((r for r in results if s in r["by_set"]), key=lambda r: -(r["by_set"][s]["recall_click"] or 0))
        print(f"\n### {s} ({rows[0]['by_set'][s]['screens'] if rows else 0} screens)\n")
        print("| combo | rule | conf | recall clickable | recall all | precision | options | ms p50 |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            m = r["by_set"][s]
            print(f"| {r['combo']} | {r['rule']} | {r['conf'] or ''} | {m['recall_click']} | {m['recall_all']} | "
                  f"{m['precision']} | {m['options']} | {m['ms_p50']} |")
    # Per-app recall for the best combos on each set, so app-level weak spots show.
    for s in sets:
        rows = sorted((r for r in results if s in r["by_set"]), key=lambda r: -(r["by_set"][s]["recall_click"] or 0))
        top = rows[: args.per_app_top]
        per = []
        for r in top:
            per.append(score([m for m in manifest if m["set"] == s], sources, tuple(r["combo"].split("+")), r["conf"],
                             r["rule"], lambda m: [m["app"]]))
        apps = sorted({a for p in per for a in p})
        print(f"\n### {s}: recall (clickable) per app\n")
        print("| app | " + " | ".join(f"{r['combo']} {r['rule']} {r['conf'] or ''}" for r in top) + " |")
        print("|---|" + "---|" * len(top))
        for a in apps:
            print(f"| {a} | " + " | ".join(str(p.get(a, {}).get("recall_click", "")) for p in per) + " |")
        for r, p in zip(top, per):
            r.setdefault("per_app", {})[s] = p
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
