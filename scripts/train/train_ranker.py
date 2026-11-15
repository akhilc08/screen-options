"""Set ranker: score every candidate on a screen jointly, from pixels and layout only (no captions, no instruction).

Each candidate is a token: its rank.py features (geometry, detector confidence, sources, OCR text stats, layout
context) plus a tiny CNN over two 32x32 crops (the box, and a 3x context window around it). A small transformer
encoder lets every box see the whole screen (toolbars, rows of peers, what's inside what), then a linear head
scores it. Loss is listwise: softmax over the screen's candidates, maximising the probability mass on the boxes
that instructions target (all targets of all examples on that screen).

Trained on other grounding sets' targets; scores for the eval sets are written for rank.py (file:<json>).

  python scripts/train/train_ranker.py --name axgpa_t --train os_atlas_macos os_atlas_linux mmbench_gui omniact weblinx \
      --eval osworld_g screenspot_pro screenspot_v2 --out runs/ranker
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from rank import FEATS, features, labels, load  # noqa: E402

DATA = ROOT / "data"
CROP = 32
MAX_CANDS = 768


def crops(screen):
    """-> uint8 (n, 2, 3, CROP, CROP): the box itself and a 3x window around it."""
    p = DATA / "public" / screen["image"]
    img = Image.open(p if p.exists() else DATA / screen["image"]).convert("RGB")     # sandbox: data/det/images/...
    W, H = img.size
    out = np.zeros((len(screen["elements"]), 2, 3, CROP, CROP), dtype=np.uint8)
    for i, e in enumerate(screen["elements"][:MAX_CANDS]):
        x, y, w, h = e["box"]
        w, h = max(w, 1), max(h, 1)
        for j, s in enumerate((1.0, 3.0)):
            cx, cy = x + w / 2, y + h / 2
            hw, hh = w * s / 2, h * s / 2
            box = (max(0, cx - hw), max(0, cy - hh), min(W, cx + hw), min(H, cy + hh))
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            out[i, j] = np.asarray(img.crop(box).resize((CROP, CROP), Image.BILINEAR)).transpose(2, 0, 1)
    return out


def featurize(screens, workers=4):
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        cs = list(ex.map(crops, screens))
    items = []
    for sc, c in zip(screens, cs):
        f = features(sc)[:MAX_CANDS]
        items.append({"f": f, "c": c[:MAX_CANDS], "y": labels(sc)[:MAX_CANDS], "image": sc["image"],
                      "n": len(sc["elements"])})
    print(f"featurized {len(screens)} screens in {time.time() - t0:.0f}s", flush=True)
    return items


class SetRanker(nn.Module):
    def __init__(self, nf, d=128, layers=3, heads=4):
        super().__init__()
        def cnn():
            return nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(64, 96, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.box_cnn, self.ctx_cnn = cnn(), cnn()
        self.feat = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        self.proj = nn.Linear(d + 192, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers) if layers else None     # 0 layers: per-box, no context
        self.head = nn.Linear(d, 1)

    def forward(self, f, c, mask):
        """f (B, N, nf) normalised features, c (B, N, 2, 3, 32, 32) float crops, mask (B, N) True = padding."""
        B, N = f.shape[:2]
        cb = self.box_cnn(c[:, :, 0].flatten(0, 1)).view(B, N, -1)
        cc = self.ctx_cnn(c[:, :, 1].flatten(0, 1)).view(B, N, -1)
        x = self.proj(torch.cat([self.feat(f), cb, cc], -1))
        if self.enc is not None:
            x = self.enc(x, src_key_padding_mask=mask)
        return self.head(x).squeeze(-1).masked_fill(mask, -1e4)


def batch(items, mu, sd, dev):
    n = max(len(it["f"]) for it in items)
    B = len(items)
    f = torch.zeros(B, n, len(FEATS))
    c = torch.zeros(B, n, 2, 3, CROP, CROP, dtype=torch.uint8)
    y = torch.zeros(B, n)
    mask = torch.ones(B, n, dtype=torch.bool)
    for i, it in enumerate(items):
        k = len(it["f"])
        f[i, :k] = torch.from_numpy((it["f"] - mu) / sd)
        c[i, :k] = torch.from_numpy(it["c"])
        y[i, :k] = torch.from_numpy(it["y"])
        mask[i, :k] = False
    return f.to(dev), (c.to(dev).float() / 255 - 0.5) / 0.25, y.to(dev), mask.to(dev)


def listwise_loss(s, y):
    """-log P(target) with P a softmax over the screen; mass on any target counts."""
    logp = F.log_softmax(s, -1)
    pos = torch.logsumexp(logp.masked_fill(y == 0, -1e4), -1)
    keep = y.sum(-1) > 0
    return -(pos[keep]).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="axgpa_t")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", default=["osworld_g", "screenspot_pro", "screenspot_v2"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no_crops", action="store_true", help="ablation: features only")
    ap.add_argument("--layers", type=int, default=3, help="0 = per-box MLP (no context)")
    ap.add_argument("--init", help="checkpoint to start from (fine-tuning)")
    ap.add_argument("--out", default="runs/ranker")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_screens = [s for t in args.train for s in load(t, args.name) if s["elements"]]
    rng = np.random.default_rng(0)
    rng.shuffle(train_screens)
    n_val = max(50, len(train_screens) // 20)
    items = featurize(train_screens)
    items = [it for it in items if it["y"].sum() > 0]
    val, train = items[:n_val], items[n_val:]
    allf = np.concatenate([it["f"] for it in train])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    model = SetRanker(len(FEATS), layers=args.layers).to(dev)
    if args.init:      # e.g. pre-trained on AX actionability; keep its feature normalisation
        ck = torch.load(args.init, weights_only=False)
        model.load_state_dict(ck["model"])
        mu, sd = ck["mu"], ck["sd"]
        print(f"init from {args.init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train) // args.bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    print(f"train {len(train)} screens, val {len(val)}, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params",
          flush=True)

    def run(its):
        model.eval()
        res = []
        with torch.no_grad():
            for i in range(0, len(its), 8):
                chunk = its[i:i + 8]
                f, c, y, m = batch(chunk, mu, sd, dev)
                if args.no_crops:
                    c = torch.zeros_like(c)
                s = model(f, c, m)
                for j, it in enumerate(chunk):
                    res.append(s[j, :len(it["f"])].float().cpu().numpy())
        return res

    def recall_at(its, scores, k=25):
        hit = n = 0
        for it, s in zip(its, scores):
            order = np.argsort(-s)[:k]
            hit += it["y"][order].sum()
            n += it["y"].sum()
        return hit / n

    best = -1
    for ep in range(args.epochs):
        model.train()
        idx = rng.permutation(len(train))
        # bucket by size so padding stays small
        idx = sorted(idx, key=lambda i: len(train[i]["f"]))
        chunks = [idx[i:i + args.bs] for i in range(0, len(idx), args.bs)]
        rng.shuffle(chunks)
        tot = 0
        for ch in chunks:
            f, c, y, m = batch([train[i] for i in ch], mu, sd, dev)
            if args.no_crops:
                c = torch.zeros_like(c)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                s = model(f, c, m)
            loss = listwise_loss(s.float(), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        r = recall_at(val, run(val))
        print(f"epoch {ep}: loss {tot / len(chunks):.3f}  val target recall@25 {r:.3f}", flush=True)
        if r > best:
            best = r
            torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "args": vars(args)}, out / "best.pt")

    ck = torch.load(out / "best.pt", weights_only=False)
    model.load_state_dict(ck["model"])
    scores = {}
    for s in args.eval:
        its = featurize(load(s, args.name))
        t0 = time.perf_counter()
        sc = run(its)
        ms = (time.perf_counter() - t0) * 1000 / len(its)
        print(f"{s}: ranker {ms:.1f} ms/screen (model only, batched)", flush=True)
        scores.update({it["image"]: [float(v) for v in x] + [-1e4] * (it["n"] - len(x)) for it, x in zip(its, sc)})
    json.dump(scores, open(out / "scores.json", "w"))
    print(f"wrote {out / 'scores.json'}", flush=True)


if __name__ == "__main__":
    main()
