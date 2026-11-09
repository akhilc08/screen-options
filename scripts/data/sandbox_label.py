"""Claude-drafted command labels for a subset of the sandbox screens (data/sandbox_ds).

  .venv/bin/python scripts/data/sandbox_label.py prep --per-split train=100 test=100 --batch 25
  .venv/bin/python scripts/data/sandbox_label.py check        # validate parts/*.jsonl, write commands.jsonl

prep picks screens round-robin across each split's apps, alternating "hard" screens (AX-poor, or 3+
detector-only icons) with ordinary ones, and writes for each one label/screens/<id>.txt (the option list
Kev sees) and label/screens/<id>.jpg (the window at 1 px per point, detector-only icons boxed in red with
their ids). Batches of screen ids go to label/batches/batch_<k>.txt; a labeler writes label/parts/batch_<k>.jsonl.
Test-split labels are drafts: they still need a human pass before they count as test truth.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/sandbox_ds"
LAB = DS / "label"
sys.path.insert(0, str(ROOT / "scripts/data"))
from label_check import problems  # noqa: E402


def load():
    screens = {json.loads(l)["id"]: json.loads(l) for l in open(DS / "screens.jsonl")}
    opts = {json.loads(l)["id"]: json.loads(l) for l in open(DS / "options.jsonl")}
    split = json.load(open(DS / "splits.json"))["apps"]
    return screens, opts, split


def n_icons(o):
    return sum(x["id"][0] == "d" for x in o["options"])


def option_text(s, o):
    lines = [f"screen {s['id']}  app: {s['app']}  title: {s.get('title')!r}  url: {s.get('url')}",
             f"mode {s['mode']}  screen {s['size']}  window at {s['frame']} (image pixel = box x,y minus window x,y)  image: {s['id']}.jpg (red boxes = detector-only icons d*)", ""]
    for x in o["options"]:
        ctx = f"  in: {' > '.join(x['context'])}" if x.get("context") else ""
        lab = f' "{x["label"]}"' if x["label"] else " (no label: look at the image)"
        lines.append(f"{x['id']}  {x['role'].removeprefix('AX').lower() or 'item'}{lab}{ctx}  box={x['box']}")
    menubar = [t for t in s["targets"] if t["source"] == "ax_menubar" and t.get("enabled", True)]
    if menubar:
        lines += ["", "menu bar:"] + [f"{t['id']}  {t['role'].removeprefix('AX').lower()} \"{t['label']}\"" for t in menubar]
    return "\n".join(lines) + "\n"


def picture(s, o, path):
    img = Image.open(ROOT / s["shot"]).convert("RGB")
    f = s["frame"]
    img = img.resize((f[2], round(img.height * f[2] / img.width)))
    d = ImageDraw.Draw(img)
    for x in o["options"]:
        if x["id"][0] == "d":
            bx, by, bw, bh = x["box"][0] - f[0], x["box"][1] - f[1], x["box"][2], x["box"][3]
            d.rectangle([bx, by, bx + bw, by + bh], outline=(255, 0, 0), width=2)
            d.text((bx + 1, max(0, by - 11)), x["id"], fill=(255, 0, 0))
    img.save(path, quality=88)


def prep(per_split, batch, seed=0):
    screens, opts, split = load()
    rng = random.Random(seed)
    chosen = []
    for sp, n in per_split.items():
        by_app = collections.defaultdict(lambda: ([], []))
        for sid, s in screens.items():
            if split[s["app"]] != sp or sid not in opts or not opts[sid]["options"]:
                continue
            hard = s["ax_poor"] or n_icons(opts[sid]) >= 3
            by_app[s["app"]][0 if hard else 1].append(sid)
        for hard, easy in by_app.values():
            rng.shuffle(hard)
            rng.shuffle(easy)
        apps = sorted(by_app)
        rng.shuffle(apps)
        picked, turn = [], 0
        while len(picked) < n and any(h or e for h, e in by_app.values()):
            for app in apps:
                hard, easy = by_app[app]
                pool = (hard or easy) if turn % 2 == 0 else (easy or hard)
                if pool and len(picked) < n:
                    picked.append(pool.pop())
            turn += 1
        chosen += [(sp, sid) for sid in picked]
    (LAB / "screens").mkdir(parents=True, exist_ok=True)
    (LAB / "batches").mkdir(exist_ok=True)
    (LAB / "parts").mkdir(exist_ok=True)
    for sp, sid in chosen:
        s, o = screens[sid], opts[sid]
        (LAB / "screens" / f"{sid}.txt").write_text(option_text(s, o))
        picture(s, o, LAB / "screens" / f"{sid}.jpg")
    rng.shuffle(chosen)
    for k in range(0, len(chosen), batch):
        (LAB / "batches" / f"batch_{k // batch}.txt").write_text("\n".join(sid for _, sid in chosen[k:k + batch]) + "\n")
    json.dump({sid: sp for sp, sid in chosen}, open(LAB / "chosen.json", "w"), indent=0)
    c = collections.Counter((sp, screens[sid]["ax_poor"]) for sp, sid in chosen)
    print(len(chosen), "screens,", (len(chosen) + batch - 1) // batch, "batches;", dict(c),
          "apps:", len({screens[s]["app"] for _, s in chosen}))


def check():
    screens, opts, _ = load()
    chosen = json.load(open(LAB / "chosen.json"))
    rows, caps, bad = [], [], 0
    for part in sorted((LAB / "parts").glob("*.jsonl")):
        for i, line in enumerate(open(part)):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                print(part.name, i, "bad json")
                continue
            sid = r.get("screen")
            if sid not in chosen:
                bad += 1
                print(part.name, i, "unknown screen", sid)
                continue
            ids = {x["id"] for x in opts[sid]["options"]} | {t["id"] for t in screens[sid]["targets"] if t["source"] == "ax_menubar"}
            if r.get("type") == "caption":
                if r.get("option") in ids and r.get("caption"):
                    caps.append({**r, "split": chosen[sid]})
                else:
                    bad += 1
                    print(part.name, i, "bad caption", r)
                continue
            try:
                p = problems(r, ids)
            except KeyError as e:
                p = [f"missing field {e}"]
            if p:
                bad += 1
                print(part.name, i, r.get("command"), "|", "; ".join(p))
                continue
            rows.append({**r, "split": chosen[sid], "labeler": "claude-draft"})
    for k, r in enumerate(rows):
        r["id"] = f"{r['screen']}-c{k}"
    with open(LAB / "commands.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(LAB / "captions.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in caps)
    n = max(len(rows), 1)
    icon = sum(any(t[0] == "d" for t in r["targets"]) for r in rows)
    print(f"{len(rows)} commands on {len({r['screen'] for r in rows})} screens, {len(caps)} icon captions, {bad} rejected")
    print("by split", dict(collections.Counter(r["split"] for r in rows)), "style", dict(collections.Counter(r["style"] for r in rows)))
    print(f"icon-only targets {icon / n:.0%}  absent {sum(r['absent'] for r in rows) / n:.0%}  "
          f"ambiguous {sum(r['ambiguous'] for r in rows) / n:.0%}  irreversible {sum(r['irreversible'] for r in rows) / n:.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "check"])
    ap.add_argument("--per-split", nargs="*", default=["train=100", "test=100"])
    ap.add_argument("--batch", type=int, default=25)
    a = ap.parse_args()
    if a.step == "prep":
        prep({k: int(v) for k, v in (x.split("=") for x in a.per_split)}, a.batch)
    else:
        check()
