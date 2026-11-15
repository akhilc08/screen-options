"""Eval screenshots that look like a screenshot the ranker trained on (perceptual dHash within --bits of 256).

Same 256-bit difference hash and 6-bit rule as rl-computer-agent's build_records.py lookalike filter (hashes are
cached in data/converted/<set>/dhash.json). Training images = the screens in data/cands/<train set>/<name>.jsonl,
i.e. exactly what train_siglip.py read. Writes the flagged eval images so every table can be re-scored without them.

  python scripts/eval/leakage.py --train os_atlas_macos os_atlas_linux mmbench_gui omniact weblinx
  -> runs/leakage.json {eval set: [image, ...]}
"""
import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def dhash(path, n=16):
    im = Image.open(path).convert("L").resize((n + 1, n), Image.BILINEAR)
    px = im.tobytes()
    return sum(1 << i for i in range(n * n) if px[(i // n) * (n + 1) + i % n] > px[(i // n) * (n + 1) + i % n + 1])


def hashes(name, images):
    cache = DATA / "converted" / name / "dhash.json"
    h = json.loads(cache.read_text()) if cache.exists() else {}
    new = 0
    for img in images:
        if img not in h:
            p = DATA / "public" / img
            try:
                h[img] = dhash(p if p.exists() else DATA / img)
            except Exception:
                h[img] = None
            new += 1
    if new:
        cache.write_text(json.dumps(h))
    return {i: h.get(i) for i in images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--train_name", default="axgpa_t")
    ap.add_argument("--eval", nargs="+", default=["osworld_g", "screenspot_pro", "screenspot_v2"])
    ap.add_argument("--name", default="axgpa_omni")
    ap.add_argument("--bits", type=int, default=6)
    args = ap.parse_args()
    train = []
    for s in args.train:
        imgs = [json.loads(l)["image"] for l in open(DATA / "cands" / s / f"{args.train_name}.jsonl")]
        train += [(s, i, h) for i, h in hashes(s, imgs).items() if h is not None]
    print(f"{len(train)} training screens hashed", flush=True)
    out = {}
    for s in args.eval:
        imgs = [json.loads(l)["image"] for l in open(DATA / "cands" / s / f"{args.name}.jsonl")]
        flagged = {}
        for img, h in hashes(s, imgs).items():
            if h is None:
                continue
            near = [(bin(h ^ th).count("1"), ts, ti) for ts, i_, th in [(t[0], t[1], t[2]) for t in train]
                    for ti in [i_] if bin(h ^ th).count("1") <= args.bits]
            if near:
                flagged[img] = sorted(near)[:3]
        out[s] = flagged
        print(f"{s}: {len(flagged)} of {len(imgs)} eval screens within {args.bits} bits of a training screen; "
              f"by training set: {sorted({n[1] for v in flagged.values() for n in v})}", flush=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "runs/leakage.json", "w"), indent=1)


if __name__ == "__main__":
    main()
