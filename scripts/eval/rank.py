"""Rank candidate lists before captioning; score target coverage at K and the captions K would cost.

Reads data/cands/<set>/<name>.jsonl (scripts/eval/candidates.py). Rankers:
  order     today's list order (OCR text first, then icons as merged)
  random    seeded shuffle
  conf      detector confidence (OCR-only text has none, so it goes last)
  prior     hand-crafted layout prior: small, near a screen edge (toolbars, sidebars), in a row/column of peers
  gbdt      LightGBM on per-box features (geometry, confidence, sources, OCR text, layout context), trained on
            the training sets' targets (task-agnostic: "which boxes do instructions point at")
  file:<p>  scores from a JSON {image: [score per element]} (e.g. the neural set ranker, scripts/train/train_ranker.py)

Metrics per eval set, over its examples: cov@K = target among the top K; full = target anywhere; caps@K = mean
icons in the top K (each needs a caption; text is labelled by OCR for free).

  python scripts/eval/rank.py --name axgpa_t --train os_atlas_macos os_atlas_linux mmbench_gui omniact weblinx \
      --eval osworld_g screenspot_pro screenspot_v2 --rankers order random conf prior gbdt
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANDS = ROOT / "data/cands"
KS = (1, 5, 10, 25, 50, 100)

FEATS = ["conf", "is_text", "src_ocr", "src_det", "x", "y", "cx", "cy", "w", "h", "logw", "logh", "loga", "aspect",
         "edge", "txt_len", "txt_words", "txt_digit", "txt_upper", "n_cands", "conf_pct", "row_peers", "col_peers",
         "near", "inside_n", "contained", "log_n"]


def load(set_name, name):
    seen, out = set(), []
    for s in map(json.loads, open(CANDS / set_name / f"{name}.jsonl")):
        if s["image"] not in seen:      # a resumed or overlapping run can append a screen twice
            seen.add(s["image"])
            out.append(s)
    return out


def features(screen):
    W, H = screen["size"]
    els = screen["elements"]
    n = len(els)
    if n == 0:
        return np.zeros((0, len(FEATS)), dtype=np.float32)
    b = np.array([e["box"] for e in els], dtype=np.float32)
    x, y, w, h = b[:, 0], b[:, 1], np.maximum(b[:, 2], 1), np.maximum(b[:, 3], 1)
    cx, cy = x + w / 2, y + h / 2
    # UI scale: the median OCR line height, so sizes mean the same on a 1x laptop and a 4K Retina screen.
    th = [e["box"][3] for e in els if e["role"] == "text" and e["box"][3] > 2]
    unit = float(np.median(th)) if len(th) >= 3 else 16.0 * max(W, H) / 1440
    conf = np.array([e["conf"] for e in els], dtype=np.float32)
    order = conf.argsort().argsort() / max(n - 1, 1)
    # Peers: same row (centre y within half a height, similar height) or same column (centre x, similar width).
    row = (np.abs(cy[:, None] - cy[None]) < 0.5 * h[:, None]) & (np.abs(np.log(h[:, None] / h[None])) < 0.4)
    col = (np.abs(cx[:, None] - cx[None]) < 0.5 * w[:, None]) & (np.abs(np.log(w[:, None] / w[None])) < 0.4)
    d = np.hypot((cx[:, None] - cx[None]) / W, (cy[:, None] - cy[None]) / H)
    near = (d < 0.05).sum(1) - 1
    x2, y2 = x + w, y + h
    inside = (x[None] >= x[:, None]) & (y[None] >= y[:, None]) & (x2[None] <= x2[:, None]) & (y2[None] <= y2[:, None])
    np.fill_diagonal(inside, False)
    rows = []
    for i, e in enumerate(els):
        t = e.get("text") or ""
        rows.append([
            conf[i], e["role"] == "text", "ocr" in e["sources"], "det" in e["sources"],
            x[i] / W, y[i] / H, cx[i] / W, cy[i] / H, w[i] / W, h[i] / H, math.log(w[i] / unit), math.log(h[i] / unit),
            math.log(w[i] * h[i] / unit ** 2), math.log(w[i] / h[i]),
            min(cx[i] / W, 1 - cx[i] / W, cy[i] / H, 1 - cy[i] / H),
            len(t), len(t.split()), sum(c.isdigit() for c in t) / max(len(t), 1),
            sum(c.isupper() for c in t) / max(len(t), 1),
            n, order[i], row[i].sum() - 1, col[i].sum() - 1, near[i], inside[i].sum(), inside[:, i].sum(), math.log(n)])
    return np.array(rows, dtype=np.float32)


def labels(screen):
    pos = {t for ex in screen["examples"] for t in ex["target_ids"]}
    return np.array([e["id"] in pos for e in screen["elements"]], dtype=np.float32)


def prior_scores(screen):
    f = features(screen)
    if len(f) == 0:
        return []
    c = {k: f[:, i] for i, k in enumerate(FEATS)}
    s = -0.5 * np.abs(c["loga"] - math.log(3)) - 8 * c["edge"] + 0.3 * np.log1p(c["row_peers"] + c["col_peers"]) \
        + 2 * c["conf"] - 0.5 * np.log1p(c["inside_n"])
    return list(s)


def train_gbdt(train_sets, name):
    import lightgbm as lgb
    X, y, g = [], [], []
    for s in train_sets:
        for sc in load(s, name):
            if not sc["elements"]:
                continue
            X.append(features(sc))
            y.append(labels(sc))
            g.append(len(sc["elements"]))
    X, y = np.concatenate(X), np.concatenate(y)
    print(f"gbdt: {len(g)} screens, {len(y)} candidates, {int(y.sum())} positives", flush=True)
    m = lgb.LGBMRanker(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(X, y.astype(int), group=g)
    imp = sorted(zip(m.feature_importances_, FEATS), reverse=True)[:10]
    print("gbdt top features:", ", ".join(f"{k} {v}" for v, k in imp), flush=True)
    return m


INST_FEATS = ["sim", "sim_pct", "sim_z", "sim_gap", "txt", "txt_pct", "txt_gap", "txt_top"]
INSTRUCTED = ("sim", "txt", "sim+txt", "gbdt_inst")


INST_TAG = ""      # which inst_feats.py output to read (--inst_tag)


def load_inst(set_name, name):
    """{image: {example id: {"sim": [...], "txt": [...]}}} from inst_feats.py."""
    p = CANDS / set_name / f"{name}.inst{INST_TAG}.jsonl"
    return {r["image"]: r["examples"] for r in map(json.loads, open(p))} if p.exists() else {}


def pct(v):
    return v.argsort().argsort() / max(len(v) - 1, 1)


def inst_features(inst):
    sim, txt = np.asarray(inst["sim"], np.float32), np.asarray(inst["txt"], np.float32)
    if len(sim) == 0:
        return np.zeros((0, len(INST_FEATS)), np.float32)
    z = (sim - sim.mean()) / (sim.std() + 1e-6)
    return np.stack([sim, pct(sim), z, sim - sim.max(), txt, pct(txt), txt - txt.max(), txt == txt.max()], 1)


def train_gbdt_inst(train_sets, name, max_ex=10):
    import lightgbm as lgb
    X, y, g = [], [], []
    for s in train_sets:
        inst = load_inst(s, name)
        for sc in load(s, name):
            if not sc["elements"] or sc["image"] not in inst:
                continue
            base = features(sc)
            for ex in sc["examples"][:max_ex]:
                if ex["id"] not in inst[sc["image"]]:
                    continue
                X.append(np.concatenate([base, inst_features(inst[sc["image"]][ex["id"]])], 1))
                pos = set(ex["target_ids"])
                y.append(np.array([e["id"] in pos for e in sc["elements"]], np.int32))
                g.append(len(sc["elements"]))
    X, y = np.concatenate(X), np.concatenate(y)
    print(f"gbdt_inst: {len(g)} examples, {len(y)} candidates, {int(y.sum())} positives", flush=True)
    m = lgb.LGBMRanker(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(X, y, group=g)
    imp = sorted(zip(m.feature_importances_, FEATS + INST_FEATS), reverse=True)[:10]
    print("gbdt_inst top features:", ", ".join(f"{k} {v}" for v, k in imp), flush=True)
    return m


def scores_for(ranker, screen, ctx, ex=None):
    n = len(screen["elements"])
    if ranker == "order":
        return [-i for i in range(n)]
    if ranker == "random":
        r = random.Random(screen["image"])
        return [r.random() for _ in range(n)]
    if ranker == "conf":
        return [e["conf"] for e in screen["elements"]]
    if ranker == "prior":
        return prior_scores(screen)
    if ranker == "gbdt":
        return list(ctx["gbdt"].predict(features(screen))) if n else []
    if ranker.startswith("file:"):
        return ctx[ranker][screen["image"]]
    inst = ctx["inst"][screen["image"]][ex["id"]]
    if ranker == "sim":
        return inst["sim"]
    if ranker == "txt":     # ties (most candidates score 0) broken by SigLIP
        return [t + 1e-3 * s for t, s in zip(inst["txt"], inst["sim"])]
    if ranker == "sim+txt":
        return list(pct(np.asarray(inst["sim"])) + pct(np.asarray(inst["txt"]) + 1e-3 * np.asarray(inst["sim"])))
    if ranker == "gbdt_inst":
        return list(ctx["gbdt_inst"].predict(np.concatenate([features(screen), inst_features(inst)], 1))) if n else []
    raise ValueError(ranker)


def evaluate(screens, ranker, ctx):
    ranks, caps = [], {k: [] for k in KS}
    per_ex = ranker in INSTRUCTED
    for sc in screens:
        groups = [[ex] for ex in sc["examples"]] if per_ex else [sc["examples"]]
        for exs in groups:
            s = scores_for(ranker, sc, ctx, exs[0] if per_ex else None)
            order = sorted(range(len(s)), key=lambda i: -s[i])
            pos = {sc["elements"][i]["id"]: r for r, i in enumerate(order)}
            icon = [sc["elements"][i]["role"] == "icon" for i in order]
            for k in KS:
                caps[k].append(sum(icon[:k]))
            for ex in exs:
                ranks.append(min((pos[t] for t in ex["target_ids"]), default=None))
    r = {"examples": len(ranks), "full": sum(x is not None for x in ranks) / len(ranks)}
    for k in KS:
        r[f"cov@{k}"] = sum(x is not None and x < k for x in ranks) / len(ranks)
        r[f"caps@{k}"] = float(np.mean(caps[k]))
    r["icons_all"] = float(np.mean([sum(e["role"] == "icon" for e in sc["elements"]) for sc in screens]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="axgpa_t")
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--train_name", help="candidate lists to train on, if not --name (e.g. axgpa_t vs axgpa_omni)")
    ap.add_argument("--eval", nargs="+", default=["osworld_g", "screenspot_pro", "screenspot_v2"])
    ap.add_argument("--rankers", nargs="+", default=["order", "random", "conf", "prior", "gbdt"])
    ap.add_argument("--out", default="runs/rank.json")
    ap.add_argument("--inst_tag", default="", help="read <name>.inst<tag>.jsonl (e.g. a fine-tuned SigLIP's)")
    ap.add_argument("--exclude", help="JSON {set: [image, ...]} of eval screens to leave out (leakage.py)")
    args = ap.parse_args()
    global INST_TAG
    INST_TAG = args.inst_tag
    ctx = {}
    if "gbdt" in args.rankers:
        ctx["gbdt"] = train_gbdt(args.train, args.train_name or args.name)
    if "gbdt_inst" in args.rankers:
        ctx["gbdt_inst"] = train_gbdt_inst(args.train, args.train_name or args.name)
    for r in args.rankers:
        if r.startswith("file:"):
            ctx[r] = json.load(open(r[5:]))
    out = {}
    drop = json.load(open(args.exclude)) if args.exclude else {}
    for s in args.eval:
        screens = [sc for sc in load(s, args.name) if sc["image"] not in drop.get(s, {})]
        if any(r in INSTRUCTED for r in args.rankers):
            ctx["inst"] = load_inst(s, args.name)
            screens = [sc for sc in screens if sc["image"] in ctx["inst"]]
        print(f"\n{s} ({len(screens)} screens)\n{'ranker':10} {'full':>6} " +
              " ".join(f"{'@' + str(k):>6}" for k in KS) + "   captions@" + "/".join(map(str, KS)) + "  (all icons)")
        for rk in args.rankers:
            r = out.setdefault(s, {})[rk] = evaluate(screens, rk, ctx)
            print(f"{rk[:10]:10} {r['full']:6.1%} " + " ".join(f"{r[f'cov@{k}']:6.1%}" for k in KS) + "   " +
                  "/".join(f"{r[f'caps@{k}']:.0f}" for k in KS) + f"  ({r['icons_all']:.0f})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
