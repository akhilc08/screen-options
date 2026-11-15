"""Leakage-filtered selector summary: runs/select/<set>/rows.jsonl minus screens in runs/leakage.json.

  python scripts/eval/select_summary.py [--exclude runs/leakage.json] -> runs/select/summary_clean.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--exclude", default=str(ROOT / "runs/leakage.json"))
ap.add_argument("--name", default="axgpa_omni")
args = ap.parse_args()
drop = json.load(open(args.exclude))
out = {}
for d in sorted((ROOT / "runs/select").iterdir()):
    if not (d / "rows.jsonl").exists():
        continue
    s = d.name
    bad = set(drop.get(s, []))
    id2img = {ex["id"]: c["image"] for c in map(json.loads, open(ROOT / "data/cands" / s / f"{args.name}.jsonl"))
              for ex in c["examples"]}
    rows = [r for r in map(json.loads, open(d / "rows.jsonl")) if id2img[r["id"]] not in bad]
    out[s] = {}
    for lname in sorted({r["list"] for r in rows}, key=lambda x: (x.split("@")[0], int(x.split("@")[1]) if "@" in x else 0)):
        rs = [r for r in rows if r["list"] == lname]
        ok = [r for r in rs if not r.get("oom")]
        cov = [r for r in rs if r["covered"]]
        med = lambda k: sorted(r[k] for r in ok)[len(ok) // 2] if ok else None
        out[s][lname] = {"n": len(rs), "n_too_long": len(rs) - len(ok),
                         "accuracy": round(100 * sum(r["correct"] for r in rs) / len(rs), 1),
                         "coverage": round(100 * len(cov) / len(rs), 1),
                         "acc_if_covered": round(100 * sum(r["correct"] for r in cov) / max(len(cov), 1), 1),
                         "median_ms": med("ms"), "median_tokens": med("tokens")}
        print(s, lname, out[s][lname])
json.dump(out, open(ROOT / "runs/select/summary_clean.json", "w"), indent=1)
