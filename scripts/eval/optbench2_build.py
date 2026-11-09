"""Option benchmark v2: screens from many apps with dense ground truth, for choosing how to build Kev's option list.

Sets (all publishable; nothing from the owner's accounts):
  groundcua  ServiceNow GroundCUA: desktop screenshots from 87 apps with EVERY element boxed by people. Targets are
             all elements; the clickable ones (Button, Menu, Input Elements, Navigation, Sidebar) are flagged.
  screen2ax  MacPaw Screen2AX-Element test split: 158 macOS screenshots with element boxes.
  sandbox    Screens from the EC2 Mac sandbox collector (scripts/sandbox/collect.py) across ~50 macOS apps and
             sites, with the AX tree walked live. Targets are the AX elements (so pixel sources are scored on how
             much of AX they recover on real Mac UI), and AX itself is recorded as a source with its live time.

Writes <out>/manifest.jsonl, <out>/images/..., and <out>/sources/ax.jsonl (sandbox only). Boxes are [x, y, w, h]
in image pixels.

  python scripts/eval/optbench2_build.py --sets groundcua,screen2ax --per_app 25 --out data/option_bench2
  python scripts/eval/optbench2_build.py --sets sandbox --sandbox_runs <run dir> ... --out data/option_bench2
"""
import argparse
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

CLICKABLE = {"Button", "Menu", "Input Elements", "Navigation", "Sidebar"}


def groundcua(out, per_app, seed):
    from huggingface_hub import HfApi, hf_hub_download
    repo = "ServiceNow/GroundCUA"
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("data/") and f.endswith(".json")]
    by_app = defaultdict(list)
    for f in files:
        by_app[f.split("/")[1]].append(f)
    rng = random.Random(seed)
    for app, fs in sorted(by_app.items()):
        for f in rng.sample(sorted(fs), min(per_app, len(fs))):
            anns = json.load(open(hf_hub_download(repo, f, repo_type="dataset")))
            if not anns:
                continue
            src = hf_hub_download(repo, "images/" + anns[0]["image_path"], repo_type="dataset")
            img = out / "images/groundcua" / anns[0]["image_path"]
            img.parent.mkdir(parents=True, exist_ok=True)
            if not img.exists():
                img.symlink_to(src)
            W, H = Image.open(src).size
            targets = [{"boxes": [[a["bbox"][0], a["bbox"][1], a["bbox"][2] - a["bbox"][0], a["bbox"][3] - a["bbox"][1]]],
                        "cls": a["category"] or "unlabelled", "clickable": a["category"] in CLICKABLE, "text": a.get("text", "")}
                       for a in anns]
            yield {"key": f"groundcua/{app}/{Path(f).stem[:16]}", "set": "groundcua", "app": app,
                   "image": str(img.relative_to(out)), "size": [W, H], "targets": targets}


def screen2ax(out):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("macpaw-research/Screen2AX-Element", "data/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    (out / "images/screen2ax").mkdir(parents=True, exist_ok=True)
    for i, r in df.iterrows():
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        name = f"images/screen2ax/{i:04d}.png"
        img.save(out / name)
        objs = r["objects"]
        targets = []
        for b, c in zip(objs["bbox"], objs["category"]):
            x1, y1, x2, y2 = [float(v) for v in b]       # corners (x2 never exceeds the image width)
            targets.append({"boxes": [[x1, y1, x2 - x1, y2 - y1]], "cls": str(c), "clickable": str(c) != "AXImage",
                            "text": ""})
        yield {"key": f"screen2ax/{i:04d}", "set": "screen2ax", "app": "macOS (mixed)", "image": name,
               "size": list(img.size), "targets": targets}


def to_px(box, frame, scale):
    x, y, w, h = box
    return [round((x - frame[0]) * scale), round((y - frame[1]) * scale), round(w * scale), round(h * scale)]


def sandbox(out, runs, per_app, seed):
    """One row per distinct screenshot, capped per app; AX elements with an on-screen frame are the targets."""
    rng = random.Random(seed)
    by_app = defaultdict(dict)
    for run in map(Path, runs):
        for line in open(run / "states.jsonl"):
            s = json.loads(line)
            key = (run.name, s["shot"])
            group = f"web:{s['url'].split('/')[2]}" if s["url"].startswith("http") else s["app"]   # cap per site, not per browser
            if key not in by_app[group]:
                by_app[group][key] = (run, s)
    ax_rows = []
    for app, shots in sorted(by_app.items()):
        for (rname, shot), (run, s) in rng.sample(sorted(shots.items()), min(per_app, len(shots))):
            src = run / shot
            name = f"images/sandbox/{rname}/{Path(shot).name}"
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            if not (out / name).exists():
                (out / name).write_bytes(src.read_bytes())
            W, H = Image.open(src).size
            frame, scale = s["frame"], s["scale"]
            boxes = [(to_px(e["frame"], frame, scale), e) for e in s["elements"] if e.get("frame")]
            boxes = [(b, e) for b, e in boxes if b[2] > 2 and b[3] > 2 and b[0] < W and b[1] < H]
            key = f"sandbox/{app}/{rname}/{Path(shot).stem}"
            site = s["url"].split("?")[0] if s["url"].startswith("http") else ""
            yield {"key": key, "set": "sandbox", "app": app if not site else f"web:{site.split('/')[2]}",
                   "image": name, "size": [W, H],
                   "targets": [{"boxes": [b], "cls": e["role"], "clickable": True, "text": e["label"]} for b, e in boxes]}
            ax_rows.append({"key": key, "ms": s["ms"]["ax_window"], "boxes": [b + [1.0, e["label"]] for b, e in boxes]})
    (out / "sources").mkdir(parents=True, exist_ok=True)
    with open(out / "sources/ax.jsonl", "a") as f:
        for r in ax_rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="groundcua,screen2ax")
    ap.add_argument("--per_app", type=int, default=25)
    ap.add_argument("--sandbox_runs", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/option_bench2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = args.sets.split(",")
    n = 0
    with open(out / "manifest.jsonl", "a") as f:
        gens = []
        if "groundcua" in sets:
            gens.append(groundcua(out, args.per_app, args.seed))
        if "screen2ax" in sets:
            gens.append(screen2ax(out))
        if "sandbox" in sets:
            gens.append(sandbox(out, args.sandbox_runs, args.per_app, args.seed))
        for g in gens:
            for row in g:
                f.write(json.dumps(row) + "\n")
                n += 1
                if n % 200 == 0:
                    print(n, row["key"], flush=True)
    print(f"{n} screens -> {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
