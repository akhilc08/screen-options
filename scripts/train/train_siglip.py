"""Fine-tune SigLIP to score a screen's candidates against an instruction (the instruction-aware ranker's core).

Per screen: crops of its candidates (context style: the element outlined in red inside a ~3x window, see
inst_feats.context_crop) and the instructions of its examples. Each instruction is scored against every candidate
on the same screen, so all the other boxes are hard negatives; the loss is a softmax over the screen's candidates
with all of the example's matched targets counted as correct. Screens with more than --max_cands candidates keep
every target and a random sample of the rest.

Training data: other grounding sets' instruction -> target pairs (data/cands/<set>/<name>.jsonl), never the eval
sets. A held-out 5% of screens tracks target recall@10.

  python scripts/train/train_siglip.py --train os_atlas_macos os_atlas_linux mmbench_gui omniact weblinx \
      --out runs/siglip/ctx
  -> runs/siglip/ctx/best (a SigLIP checkpoint for inst_feats.py --model ... --crop context)
"""
import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from inst_feats import MODEL, crop, vec  # noqa: E402
from rank import load  # noqa: E402

DATA = ROOT / "data"


def screen_items(sets, name):
    out = []
    for s in sets:
        for sc in load(s, name):
            exs = [ex for ex in sc["examples"] if ex["target_ids"] and ex["instruction"]]
            if sc["elements"] and exs:
                out.append({"image": sc["image"], "elements": sc["elements"], "examples": exs})
    return out


def prepare(item, style, max_cands, max_ex, rng):
    """-> (crops, instructions, positive index sets) for one screen, candidates subsampled around the targets."""
    p = DATA / "public" / item["image"]
    img = Image.open(p if p.exists() else DATA / item["image"]).convert("RGB")
    exs = item["examples"] if len(item["examples"]) <= max_ex else rng.sample(item["examples"], max_ex)
    pos = {int(t[1:]) for ex in exs for t in ex["target_ids"]}
    rest = [i for i in range(len(item["elements"])) if i not in pos]
    keep = sorted(pos) + rng.sample(rest, min(len(rest), max(0, max_cands - len(pos))))
    where = {i: k for k, i in enumerate(keep)}
    crops = [crop(img, item["elements"][i]["box"], style) for i in keep]
    targets = [[where[int(t[1:])] for t in ex["target_ids"] if int(t[1:]) in where] for ex in exs]
    return crops, [ex["instruction"] for ex in exs], targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--name", default="axgpa_t")
    ap.add_argument("--crop", default="context", choices=("tight", "context"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_cands", type=int, default=160)
    ap.add_argument("--max_ex", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4, help="screens per optimiser step")
    ap.add_argument("--model", default=MODEL, help="base checkpoint, e.g. google/siglip-base-patch16-384")
    ap.add_argument("--freeze", type=int, default=0, help="freeze embeddings + this many lower layers per tower (0: none)")
    ap.add_argument("--loss", default="listwise", choices=("listwise", "sigmoid"),
                    help="listwise: softmax over the screen's candidates (ours); sigmoid: pointwise per pair")
    ap.add_argument("--out", default="runs/siglip/ctx")
    args = ap.parse_args()
    from transformers import AutoModel, AutoProcessor
    rng = random.Random(0)
    torch.manual_seed(0)
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).cuda()
    model.gradient_checkpointing_enable() if hasattr(model, "gradient_checkpointing_enable") else None
    items = screen_items(args.train, args.name)
    rng.shuffle(items)
    n_val = max(100, len(items) // 20)
    val, train = items[:n_val], items[n_val:]
    print(f"train {len(train)} screens / {sum(len(i['examples']) for i in train)} examples, val {len(val)}", flush=True)
    if args.freeze:    # big models: freeze embeddings and the lower `freeze` layers of each tower to fit Adam in memory
        import re
        for n, p in model.named_parameters():
            m = re.search(r"layers\.(\d+)\.", n)
            if "embeddings" in n or (m and int(m.group(1)) < args.freeze):
                p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.0f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, foreach=False)
    total = args.epochs * len(train) // args.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total + 1, pct_start=0.05)
    pool = ThreadPoolExecutor(3)
    out = Path(args.out)

    def score(crops, insts):
        px = proc(images=crops, return_tensors="pt")["pixel_values"].cuda()
        tok = proc(text=insts, padding="max_length", max_length=min(64, model.config.text_config.max_position_embeddings), truncation=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ie = F.normalize(vec(model.get_image_features(pixel_values=px)).float(), dim=-1)
            te = F.normalize(vec(model.get_text_features(**tok)).float(), dim=-1)
        return te @ ie.T * model.logit_scale.exp()

    def evaluate():
        model.eval()
        hit = n = 0
        with torch.no_grad():
            for it in val:
                crops, insts, tg = prepare(it, args.crop, 10_000, args.max_ex, random.Random(1))
                s = score(crops, insts)
                for k, t in enumerate(tg):
                    if t:
                        top = set(s[k].topk(min(10, s.shape[1])).indices.tolist())
                        hit += bool(top & set(t))
                        n += 1
        model.train()
        return hit / max(n, 1)

    print(f"before: val recall@10 {evaluate():.3f}", flush=True)
    best, step = -1, 0
    for ep in range(args.epochs):
        rng.shuffle(train)
        t0 = time.time()
        futs = [pool.submit(prepare, it, args.crop, args.max_cands, args.max_ex, random.Random(ep * 10**6 + i))
                for i, it in enumerate(train[:6])]
        tot = cnt = 0
        for i in range(len(train)):
            crops, insts, tg = futs[i].result()
            if i + 6 < len(train):
                futs.append(pool.submit(prepare, train[i + 6], args.crop, args.max_cands, args.max_ex,
                                        random.Random(ep * 10**6 + i + 6)))
            futs[i] = None
            s = score(crops, insts)
            if args.loss == "sigmoid":      # pointwise: SigLIP's own pairwise sigmoid over every (instruction, box)
                y = torch.zeros_like(s)
                for k, t in enumerate(tg):
                    y[k, t] = 1.0
                keep = [k for k, t in enumerate(tg) if t]
                losses = [F.binary_cross_entropy_with_logits(s[keep] + model.logit_bias, y[keep], reduction="sum")
                          / max(len(keep), 1)] if keep else []
            else:
                logp = F.log_softmax(s, -1)
                losses = [-torch.logsumexp(logp[k, t], 0) for k, t in enumerate(tg) if t]
            if not losses:
                continue
            loss = torch.stack(losses).mean() / args.accum
            loss.backward()
            tot += loss.item() * args.accum
            cnt += 1
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                sched.step()
                step += 1
            if (i + 1) % 500 == 0:
                print(f"epoch {ep} {i + 1}/{len(train)}: loss {tot / cnt:.3f}  {time.time() - t0:.0f}s", flush=True)
                tot = cnt = 0
        r = evaluate()
        print(f"epoch {ep}: val recall@10 {r:.3f}", flush=True)
        if r > best:
            best = r
            model.save_pretrained(out / "best")
            proc.save_pretrained(out / "best")
    print(f"best val recall@10 {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
