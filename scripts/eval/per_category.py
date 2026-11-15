"""Per-category coverage of the frozen ranker (leakage-filtered), for the supplement.

ScreenSpot-Pro: application group x text/icon; ScreenSpot-v2: platform x text/icon (benchmark annotations);
all three: target area as a share of the screen. Columns: n, listed, cov@10, cov@50, UI-Venus-1.5-8B click@1.

  python scripts/eval/per_category.py --inst_tag _384 -> runs/per_category.json
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "data/public"


def meta_sspro():
    m = {}
    for f in glob.glob(str(PUB / "screenspot_pro/annotations/*.json")):
        for a in json.load(open(f)):
            m["sspro-" + a["id"]] = (a["group"], a["ui_type"])
    return m


def meta_ssv2():
    m = {}
    for p in ("web", "mobile", "desktop"):
        for i, a in enumerate(json.load(open(PUB / f"screenspot_v2/screenspot_{p}_v2.json"))):
            m[f"ssv2-{p}-{i}"] = (p, a["data_type"], a["instruction"])
    return m


def area_bin(a):
    return "area < 0.01%" if a < 1e-4 else "0.01-0.1%" if a < 1e-3 else "area >= 0.1%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst_tag", default="_384")
    ap.add_argument("--name", default="axgpa_omni")
    ap.add_argument("--exclude", default=str(ROOT / "runs/leakage.json"))
    args = ap.parse_args()
    drop = json.load(open(args.exclude))
    metas = {"screenspot_pro": meta_sspro(), "screenspot_v2": meta_ssv2()}
    out = {}
    for s in ("osworld_g", "screenspot_pro", "screenspot_v2"):
        d = ROOT / "data/cands" / s
        inst = {c["image"]: c["examples"] for c in map(json.loads, open(d / f"{args.name}.inst{args.inst_tag}.jsonl"))}
        venus = {r["id"]: bool(r.get("hit")) for r in map(json.loads, open(ROOT / f"runs/vlm/venus-8b/{s}/rows.jsonl"))}
        gt = {}
        if s == "osworld_g":
            for a in json.load(open(PUB / "osworld_g/annotations.json")):
                ld = lambda v: json.loads(v) if isinstance(v, str) else v
                b = ld(a["box_coordinates"]) if a["box_type"] == "bbox" else None
                W, H = ld(a["image_size"])
                gt["osworldg-" + a["id"]] = b[2] * b[3] / (W * H) if b else None
        groups = defaultdict(list)
        mism = 0
        for c in map(json.loads, open(d / f"{args.name}.jsonl")):
            if c["image"] in drop.get(s, []):
                continue
            W, H = c["size"]
            for ex in c["examples"]:
                sim = inst.get(c["image"], {}).get(ex["id"], {}).get("sim")
                if sim is None:
                    continue
                t = {int(x[1:]) for x in ex["target_ids"]}
                order = sorted(range(len(sim)), key=lambda i: -sim[i])
                rank = min((order.index(i) for i in t), default=None)
                row = (bool(t), rank is not None and rank < 10, rank is not None and rank < 50, venus.get(ex["id"]))
                keys = ["all"]
                if s == "screenspot_pro":
                    g, u = metas[s][ex["id"]]
                    keys += [f"group {g}", f"type {u}"]
                elif s == "screenspot_v2":
                    p, u, ins = metas[s][ex["id"]]
                    mism += ins.strip() != ex["instruction"].strip()
                    keys += [f"platform {p}", f"type {u}"]
                if s == "osworld_g":
                    a = gt.get(ex["id"])
                    if a is not None:
                        keys.append(area_bin(a))
                for k in keys:
                    groups[k].append(row)
        out[s] = {}
        print(f"\n{s}" + (f"  (instruction mismatches vs annotations: {mism})" if s == "screenspot_v2" else ""))
        for k, rs in sorted(groups.items(), key=lambda kv: (kv[0] != "all", kv[0])):
            n = len(rs)
            v = [r[3] for r in rs if r[3] is not None]
            out[s][k] = {"n": n, "listed": round(100 * sum(r[0] for r in rs) / n, 1),
                         "cov@10": round(100 * sum(r[1] for r in rs) / n, 1), "cov@50": round(100 * sum(r[2] for r in rs) / n, 1),
                         "venus8b@1": round(100 * sum(v) / len(v), 1) if v else None}
            print(f"  {k:22s} {out[s][k]}")
    json.dump(out, open(ROOT / "runs/per_category.json", "w"), indent=1)


if __name__ == "__main__":
    main()
