"""Re-build pixel option lists for the public grounding sets and give icons better labels, cheaply.

The converted sets in rl-computer-agent (data/converted/<set>/screens.jsonl) already hold Apple Vision OCR
for every screen (elements tagged "ocr"), so only the detector and the icon captions are re-run here, on any
GPU box. No Mac needed.

Stages:
  detect    OCR (reused) + a detector + Florence-2 crop captions, merged as at serving (sources.merge, no AX).
            -> data/relabel/<set>/<det>@<size>/screens.jsonl
  variants  label variants built from those screens with no extra model (a few ms per screen):
              crop      Florence-2 crop caption only (today's labels)
              near      + the short OCR text right next to the icon, if any
              ctx       + where it is on screen and the nearest section header above it
              near_ctx  both
            plus "orig" = the existing converted list (OmniParser v2 @1280), for the baseline.
            -> data/relabel/<set>/<det>@<size>/<variant>/{examples.jsonl, records.jsonl}
  records   Kev records for the fixed eval samples (data/records_s ids), same format as build_records.py.

  python scripts/eval/relabel.py detect --set osworld_g --detector gpa --imgsz 640
  python scripts/eval/relabel.py variants --set osworld_g --detector gpa --imgsz 640 --rlca ../rlca
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from screen_options.sources import merge  # noqa: E402

DATA = ROOT / "data"
RECORDS_S = {"osworld_g": "eval_osworldg", "screenspot_pro": "eval_sspro", "screenspot_v2": "eval_ssv2",
             "mmbench_gui": "eval_mmbench", "omniact": "eval_omniact"}
PUBLIC_ACTIONS = {"click": "click", "double_click": "click", "right_click": "click", "type": "type"}
MAX_OPTIONS = 255
VARIANTS = ("crop", "near", "ctx", "near_ctx")


def sample_ids(name):
    return {json.loads(l)["_meta"]["id"] for l in open(DATA / "records_s" / f"{RECORDS_S[name]}.jsonl")}


def sample_examples(name):
    ids = sample_ids(name)
    return [ex for ex in map(json.loads, open(DATA / "converted" / name / "examples.jsonl")) if ex["id"] in ids]


def out_dir(args):
    return DATA / "relabel" / args.set / f"{args.detector}@{args.imgsz}"


# ---------------------------------------------------------------- detect (GPU)

def detect(args):
    from PIL import Image
    from screen_options.detect import Captioner, Detector
    examples = sample_examples(args.set)
    images = list(dict.fromkeys(ex["image"] for ex in examples))
    old = {s["image"]: s for s in map(json.loads, open(DATA / "converted" / args.set / "screens.jsonl"))}
    out = out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "screens.jsonl").exists():
        done = {json.loads(l)["image"] for l in open(out / "screens.jsonl")}
    det, cap = Detector(model=args.weights or args.detector, imgsz=args.imgsz, tile=args.tile), Captioner()
    print(f"{args.set}: {len(examples)} examples on {len(images)} screens, {len(done)} done", flush=True)
    with open(out / "screens.jsonl", "a") as f:
        for i, rel in enumerate(r for r in images if r not in done):
            img = Image.open(DATA / "public" / rel).convert("RGB")
            frame = (0, 0, img.width, img.height)
            ocr = [{"box": tuple(e["box"]), "text": e["label"]} for e in old[rel]["elements"] if "ocr" in e["sources"]]
            t0 = time.perf_counter()
            boxes = det.boxes(img, frame, 1.0)
            t1 = time.perf_counter()
            items, need = merge([], ocr, boxes, frame)
            px = [(it.box[0], it.box[1], it.box[0] + it.box[2], it.box[1] + it.box[3]) for it in (items[j] for j in need)]
            for j, text in zip(need, cap.captions(img, px) if px else []):
                items[j].label = text
            t2 = time.perf_counter()
            els = [{"id": f"e{k}", "role": it.role, "label": it.label, "box": list(it.box), "sources": sorted(it.sources)}
                   for k, it in enumerate(items)]
            f.write(json.dumps({"image": rel, "size": [img.width, img.height], "elements": els,
                                "ms": {"detect": round((t1 - t0) * 1000), "caption": round((t2 - t1) * 1000)}}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{args.set} {args.detector}@{args.imgsz} {i}: {len(els)} options, "
                      f"{round((t1 - t0) * 1000)} + {round((t2 - t1) * 1000)} ms", flush=True)


def recaption(args):
    """Re-label the icons of an existing <det>@<size> list with another captioner (e.g. the AX-trained labeller).
    The boxes and OCR are untouched, so any change in Kev's score comes from the labels alone.
    -> data/relabel/<set>/<det>@<size>+<tag>/screens.jsonl"""
    from PIL import Image
    from screen_options.detect import Captioner
    src = out_dir(args)
    dst = src.parent / f"{src.name}+{args.tag}"
    dst.mkdir(parents=True, exist_ok=True)
    cap = Captioner(model_dir=Path(args.captioner).expanduser())
    with open(dst / "screens.jsonl", "w") as f:
        for i, s in enumerate(map(json.loads, open(src / "screens.jsonl"))):
            icons = [e for e in s["elements"] if e["role"] == "icon"]
            if icons:
                img = Image.open(DATA / "public" / s["image"]).convert("RGB")
                px = [(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in (e["box"] for e in icons)]
                t0 = time.perf_counter()
                for e, text in zip(icons, cap.captions(img, px, max_new_tokens=24)):
                    role, sep, label = text.partition(": ")
                    # Keep role "icon" so the label variants still treat it as a pixel-found icon; the
                    # predicted role goes into the label text Kev reads.
                    e["label"] = f"{role} {label}".strip() if sep else text
                s["ms"] = {**s.get("ms", {}), "caption": round((time.perf_counter() - t0) * 1000)}
            f.write(json.dumps(s) + "\n")
            if i % 100 == 0:
                print(f"{args.set} {dst.name} {i}", flush=True)


# ---------------------------------------------------------------- label variants (CPU, no model)

def words(t):
    return len(t.split())


def gap(a, b):
    """Edge-to-edge distance between two (x, y, w, h) boxes (0 if they touch or overlap)."""
    dx = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return (dx * dx + dy * dy) ** 0.5


def v_overlap(a, b):
    return min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])


def h_overlap(a, b):
    return min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])


def near_text(icon, texts):
    """Short text directly beside (same row) or under the icon, within a couple of icon heights."""
    h = max(icon[3], 8)
    best = None
    for t in texts:
        b = t["box"]
        same_row = v_overlap(icon, b) > 0.4 * min(icon[3], b[3])
        below = h_overlap(icon, b) > 0.3 * min(icon[2], b[2]) and b[1] >= icon[1] + icon[3] - 2
        d = gap(icon, b)
        if (same_row and d <= 1.5 * h) or (below and d <= 1.0 * h):
            if best is None or d < best[0]:
                best = (d, t["label"])
    return best[1] if best else None


def region(box, W, H):
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    v = "top" if cy < H * 0.12 else "bottom" if cy > H * 0.88 else "middle"
    hz = "left" if cx < W * 0.2 else "right" if cx > W * 0.8 else "center"
    return "top bar" if v == "top" and hz == "center" else f"{v} {hz}".replace("middle center", "center")


def header_above(box, texts, W, H):
    """Nearest short text line above the box whose columns overlap it: a section or panel title."""
    best = None
    for t in texts:
        b = t["box"]
        if b[1] + b[3] > box[1] or words(t["label"]) > 4:
            continue
        dy = box[1] - (b[1] + b[3])
        if dy > 0.2 * H or h_overlap((box[0] - 0.05 * W, 0, box[2] + 0.1 * W, 1), b) <= 0:
            continue
        if best is None or dy < best[0]:
            best = (dy, t["label"])
    return best[1] if best else None


def with_variant(screen, variant):
    W, H = screen["size"]
    texts = [e for e in screen["elements"] if "ocr" in e["sources"] and words(e["label"]) <= 6]
    out = []
    for e in screen["elements"]:
        e = dict(e)
        if e["role"] == "icon" and variant != "crop":
            if "near" in variant:
                t = near_text(e["box"], texts)
                if t:
                    e["label"] = f'{e["label"]}, next to "{t}"'
            if "ctx" in variant:
                hdr = header_above(e["box"], texts, W, H)
                e["context"] = [region(e["box"], W, H)] + ([hdr] if hdr else [])
        out.append(e)
    return out


def match(elements, gt_box=None, gt_point=None, max_area_ratio=4.0):
    """Same rule as rl-computer-agent's convert_public.match: centre in the answer box or IoU > 0.3, not a container."""
    from screen_options.sources import iou
    hits = []
    for e in elements:
        b = e["box"]
        if gt_box is not None:
            c = (b[0] + b[2] / 2, b[1] + b[3] / 2)
            inside = gt_box[0] <= c[0] <= gt_box[0] + gt_box[2] and gt_box[1] <= c[1] <= gt_box[1] + gt_box[3]
            s = iou(b, gt_box)
            if (inside or s > 0.3) and b[2] * b[3] <= max_area_ratio * max(gt_box[2] * gt_box[3], 1):
                hits.append((-s, e["id"]))
        elif b[0] <= gt_point[0] <= b[0] + b[2] and b[1] <= gt_point[1] <= b[1] + b[3]:
            hits.append((b[2] * b[3], e["id"]))
    return [i for _, i in sorted(hits)]


def write_variant(name, screens, examples, d, rlca):
    sys.path.insert(0, str(Path(rlca).resolve() / "src"))
    from rlca.records import record
    d.mkdir(parents=True, exist_ok=True)
    n = hit = 0
    with open(d / "examples.jsonl", "w") as fe, open(d / "records.jsonl", "w") as fr:
        for ex in examples:
            s = screens.get(ex["image"])
            action = PUBLIC_ACTIONS.get(ex.get("action"))
            if s is None or action is None:
                continue
            ids = match(s, gt_box=ex.get("gt_box"), gt_point=None if ex.get("gt_box") else ex.get("gt_point"))
            opts = s[: MAX_OPTIONS - 1]
            keep = {o["id"] for o in opts}
            targets = [t for t in ids if t in keep]
            fe.write(json.dumps({**ex, "target_ids": ids, "n_options": len(s)}) + "\n")
            app = ex.get("app") or ex.get("platform") or name
            rec = record(app, ex["instruction"], opts, targets, action=action,
                         text_source=("user" if action == "type" else None),
                         meta={"source": name, "id": ex["id"], "split": ex["split"],
                               **({"platform": ex["platform"]} if ex.get("platform") else {})})
            fr.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            hit += bool(targets)
    return n, hit


def variants(args):
    examples = sample_examples(args.set)
    base = out_dir(args)
    if args.list:
        base = base.parent / f"{base.name}+{args.list}"
    report = {}
    if args.orig:
        old = {s["image"]: s["elements"] for s in map(json.loads, open(DATA / "converted" / args.set / "screens.jsonl"))}
        report["orig"] = write_variant(args.set, old, examples, DATA / "relabel" / args.set / "orig", args.rlca)
    screens = [s for s in map(json.loads, open(base / "screens.jsonl")) if s.get("size")]   # hung screens have none
    for v in VARIANTS:
        sv = {s["image"]: with_variant(s, v) for s in screens}
        report[v] = write_variant(args.set, sv, examples, base / v, args.rlca)
    for v, (n, hit) in report.items():
        print(f"{args.set} {args.detector}@{args.imgsz} {v}: {n} examples, target in list {hit / max(n, 1):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("detect", "recaption", "variants"))
    ap.add_argument("--set", required=True, choices=RECORDS_S)
    ap.add_argument("--detector", default="gpa")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--weights", default="", help="detect: local .pt for --detector (which then only names the list)")
    ap.add_argument("--tile", type=float, default=0, help="detect: tile images longer than tile*imgsz (0 = never)")
    ap.add_argument("--list", default="", help="variants: read <det>@<size>+<list> instead (a recaptioned list)")
    ap.add_argument("--captioner", default="~/.cache/rlca/ax-labeler", help="recaption: model dir")
    ap.add_argument("--tag", default="ax", help="recaption: suffix for the output list")
    ap.add_argument("--rlca", default=str(ROOT.parent / "rl-computer-agent"), help="rl-computer-agent checkout (for Kev's record format)")
    ap.add_argument("--orig", action="store_true", help="also write the existing converted list as the 'orig' baseline")
    args = ap.parse_args()
    {"detect": detect, "recaption": recaption, "variants": variants}[args.stage](args)


if __name__ == "__main__":
    main()
