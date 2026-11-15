"""Option-list coverage without a chooser: is the target on the list, and how long is the list.

Reads every data/relabel/<set>/<list>/crop/examples.jsonl (target_ids are matched on the full, uncapped list;
element ids e<i> are list positions, OCR text first, then icons). For each list:
  full      target matched anywhere on the uncapped list
  @K        target among the first K options (K = 50, 100, 150, 254 = today's cap in relabel.py)
  options   median / p90 uncapped list length
  det ms    median detection + caption time per screen as logged by relabel.py detect

  python3 scripts/eval/coverage.py [--sets osworld_g screenspot_pro screenspot_v2] [--json out.json]
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/relabel"
KS = (50, 100, 150, 254)


def summarize(d):
    ex = [json.loads(l) for l in open(d / "crop/examples.jsonl")]
    first = [min((int(t[1:]) for t in e["target_ids"]), default=None) for e in ex]
    n = [e["n_options"] for e in ex]
    r = {"examples": len(ex), "full": sum(f is not None for f in first) / len(ex)}
    for k in KS:
        r[f"@{k}"] = sum(f is not None and f < k for f in first) / len(ex)
    r["options"] = statistics.median(n)
    r["options_p90"] = sorted(n)[int(0.9 * len(n))]
    ms = [s["ms"] for s in map(json.loads, open(d / "screens.jsonl")) if s.get("ms")]
    if ms:
        r["det_ms"] = statistics.median(m.get("detect", 0) for m in ms)
        r["cap_ms"] = statistics.median(m.get("caption", 0) for m in ms)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["osworld_g", "screenspot_pro", "screenspot_v2"])
    ap.add_argument("--json")
    args = ap.parse_args()
    out = {}
    for s in args.sets:
        print(f"\n{s}\n{'list':22} {'full':>6} " + " ".join(f"{'@' + str(k):>6}" for k in KS) +
              f" {'opts':>5} {'p90':>5} {'det ms':>7} {'cap ms':>7}")
        for d in sorted(p for p in (DATA / s).iterdir() if (p / "crop/examples.jsonl").exists() and "+" not in p.name):
            r = out.setdefault(s, {})[d.name] = summarize(d)
            print(f"{d.name:22} {r['full']:6.1%} " + " ".join(f"{r[f'@{k}']:6.1%}" for k in KS) +
                  f" {r['options']:5.0f} {r['options_p90']:5} {r.get('det_ms', 0):7.0f} {r.get('cap_ms', 0):7.0f}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
