"""Grounding models as option-list rankers, alone and combined with ours, on the same candidate lists.

A grounder (UI-Venus, GUI-Actor, ...) outputs one click point (runs/vlm/<model>/<set>/rows.jsonl, from
rl-computer-agent's vlm_grounding.py on the same 500 examples per set). Over our candidates (data/cands/<set>/
<name>.jsonl) that point becomes a ranking: candidates containing it first (smallest first), then by distance from
it. "combo" puts the grounder's containing candidate first and then our instruction ranker's order (sim from
<name>.inst<tag>.jsonl), i.e. the grounder's pick plus our alternatives for a selector model to recover with.
Coverage@K = fraction of all examples whose target is among the first K (unlisted targets count as misses).

  python scripts/eval/grounder_combo.py --inst_tag _384 --models venus-2b venus-8b gui-actor-2b
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KS = (1, 3, 5, 10, 25, 50)


def contains(b, p):
    return b[0] <= p[0] <= b[0] + b[2] and b[1] <= p[1] <= b[1] + b[3]


def by_point(els, p):
    def key(i):
        x, y, w, h = els[i]["box"]
        inside = contains(els[i]["box"], p)
        d = ((x + w / 2 - p[0]) ** 2 + (y + h / 2 - p[1]) ** 2) ** 0.5
        return (0, w * h) if inside else (1, d)
    return sorted(range(len(els)), key=key)


def cover(order, targets, k):
    return bool(set(order[:k]) & targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["osworld_g", "screenspot_pro", "screenspot_v2"])
    ap.add_argument("--name", default="axgpa_omni")
    ap.add_argument("--inst_tag", default="_384")
    ap.add_argument("--models", nargs="+", default=["venus-2b", "venus-8b", "gui-actor-2b", "qwen3vl-2b"])
    ap.add_argument("--exclude", help="JSON {set: [image, ...]} of eval screens to leave out (leakage.py)")
    args = ap.parse_args()
    drop = json.load(open(args.exclude)) if args.exclude else {}
    out = {}
    for s in args.sets:
        d = ROOT / "data/cands" / s
        cands = {c["image"]: c for c in map(json.loads, open(d / f"{args.name}.jsonl"))}
        inst = {c["image"]: c for c in map(json.loads, open(d / f"{args.name}.inst{args.inst_tag}.jsonl"))}
        exs = []
        for img, c in cands.items():
            if img in drop.get(s, {}):
                continue
            for ex in c["examples"]:
                sim = inst.get(img, {}).get("examples", {}).get(ex["id"], {}).get("sim")
                if sim is None:
                    continue
                ours = sorted(range(len(sim)), key=lambda i: -sim[i])
                exs.append((ex["id"], c["elements"], {int(t[1:]) for t in ex["target_ids"]}, ours))
        rows = {m: {r["id"]: r for r in map(json.loads, open(ROOT / "runs/vlm" / m / s / "rows.jsonl"))}
                for m in args.models if (ROOT / "runs/vlm" / m / s / "rows.jsonl").exists()}
        print(f"\n{s}: {len(exs)} examples   coverage@{'/'.join(map(str, KS))}")
        res = {"ours": [sum(cover(o, t, k) for _, _, t, o in exs) / len(exs) for k in KS]}
        for m, R in rows.items():
            g, cb, raw, n = [0] * len(KS), [0] * len(KS), 0, 0
            for eid, els, t, ours in exs:
                r = R.get(eid)
                n += 1
                if not r or not r.get("point"):
                    order_g, order_c = [], ours
                else:
                    raw += bool(r.get("hit"))
                    order_g = by_point(els, r["point"])
                    first = [i for i in order_g[:1] if contains(els[i]["box"], r["point"])]
                    order_c = first + [i for i in ours if i not in first]
                for j, k in enumerate(KS):
                    g[j] += cover(order_g, t, k)
                    cb[j] += cover(order_c, t, k)
            res[m] = [x / n for x in g]
            res[f"{m}+ours"] = [x / n for x in cb]
            res[f"{m} raw@1"] = [raw / n]
        for k, v in res.items():
            print(f"  {k:22s} " + " ".join(f"{100 * x:5.1f}" for x in v))
        out[s] = res
    (ROOT / "runs").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / f"runs/grounder_combo{args.inst_tag}{'_clean' if args.exclude else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
