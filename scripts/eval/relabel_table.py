"""One table for the relabel runs: coverage, Kev's pick rate when the target is listed, and overall accuracy.

Reads Kev's rows (kev.benchmark output, pulled from the boxes into --runs) and the records each run used.
Element accuracy follows score_kev.py: the top option counts if it is any acceptable target ("none" when the
target isn't listed).

  python3 scripts/eval/relabel_table.py --runs data/relabel_runs [--json out.json]
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/relabel"
SETS = {"osworld_g": "OSWorld-G", "screenspot_pro": "ScreenSpot-Pro", "screenspot_v2": "ScreenSpot-v2"}


def score(rows_path, records_path):
    recs = {r["_meta"]["id"]: r for r in map(json.loads, open(records_path))}
    n = hit = listed = listed_hit = 0
    opts = []
    for row in json.load(open(rows_path)):
        if row.get("question") != "element" or row.get("variant", "clean") != "clean" or row["id"] not in recs:
            continue
        q = recs[row["id"]]["questions"]["element"]
        ok = set(q.get("target") or [q["label"]])
        p = row.get("logits") or row.get("p")
        top = row["keys"][max(range(len(p)), key=p.__getitem__)]
        good = top in ok
        n += 1
        hit += good
        opts.append(len(q["criteria"]) - 1)
        if q["label"] != "none":
            listed += 1
            listed_hit += good
    return {"n": n, "acc": hit / n, "in_list": listed / n, "pick_when_listed": listed_hit / max(listed, 1),
            "options": statistics.median(opts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "data/relabel_runs"), help="local copy of ~/kev/runs/relabel")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    runs = Path(args.runs)
    out = []
    for rows in sorted(runs.glob("*/**/rows.json")):
        rel = rows.parent.relative_to(runs)           # <set>/orig  or  <set>/<list>/<variant>
        parts = rel.parts
        s = parts[0]
        lst, var = ("omni_v2@1280", "crop") if parts[1] == "orig" else (parts[1], parts[2])
        rec = DATA / s / "orig" / "records.jsonl" if parts[1] == "orig" else DATA / s / lst / var / "records.jsonl"
        if not rec.exists():
            print(f"missing records for {rel}")
            continue
        out.append({"set": s, "list": lst, "labels": var, **score(rows, rec)})
    for s, name in SETS.items():
        print(f"\n{name}")
        print(f"{'list':<18}{'labels':<10}{'in list':>9}{'pick|listed':>13}{'accuracy':>10}{'options':>9}")
        for r in sorted((r for r in out if r["set"] == s), key=lambda r: (r["list"], r["labels"])):
            print(f"{r['list']:<18}{r['labels']:<10}{r['in_list']:>9.1%}{r['pick_when_listed']:>13.1%}{r['acc']:>10.1%}{r['options']:>9.0f}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
