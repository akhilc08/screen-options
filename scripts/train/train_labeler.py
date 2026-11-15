"""Fine-tune OmniParser's Florence-2 icon captioner to name icons the way accessibility does.

Input is the same 64x64 crop the captioner sees at serving (5 image tokens, no resize), so the trained
labeller costs exactly what today's captioner costs. Target text is "<role>: <AX label>" from
scripts/data/ax_label_pairs.py, e.g. "button: Go back".

  python scripts/train/train_labeler.py --data data/ax_labels --out ~/.cache/rlca/ax-labeler [--epochs 2]
  python scripts/train/train_labeler.py --data data/ax_labels --eval_only --model ~/.cache/rlca/omniparser-caption

Eval (val and test apps, never trained on): exact match of the label part (case-insensitive), token F1, and
role accuracy. The untrained captioner is scored the same way first, so the gain is measured on the same pairs.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CAPTION_DIR = Path.home() / ".cache/rlca/omniparser-caption"
CROP_TOKENS = 5
PROMPT = "<CAPTION>"


def load(model_dir, dtype):
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_dir)
    proc.num_image_tokens = CROP_TOKENS - proc.num_additional_image_tokens
    model = Florence2ForConditionalGeneration.from_pretrained(model_dir, dtype=dtype).cuda()
    return proc, model


class Pairs(Dataset):
    def __init__(self, root, split, limit=0, seed=0):
        self.root = Path(root)
        self.rows = [json.loads(l) for l in open(self.root / f"{split}.jsonl")]
        if limit and len(self.rows) > limit:
            self.rows = random.Random(seed).sample(self.rows, limit)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return Image.open(self.root / "crops" / f"{r['id']}_crop.png").convert("RGB"), r["text"]


def collate(proc):
    def f(batch):
        imgs, texts = zip(*batch)
        enc = proc(text=[PROMPT] * len(imgs), images=list(imgs), return_tensors="pt", do_resize=False)
        lab = proc.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=32).input_ids
        lab[lab == proc.tokenizer.pad_token_id] = -100
        enc["labels"] = lab
        enc["texts"] = list(texts)
        return enc
    return f


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def split_text(t):
    role, _, label = t.partition(": ")
    return (role, label) if label else ("", t)


def f1(a, b):
    a, b = norm(a).split(), norm(b).split()
    if not a or not b:
        return float(a == b)
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    if not common:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


@torch.inference_mode()
def evaluate(proc, model, ds, dtype, batch=128, show=0):
    model.eval()
    em = tf = ra = n = 0
    examples = []
    t0 = time.perf_counter()
    for enc in DataLoader(ds, batch_size=batch, collate_fn=collate(proc), num_workers=4):
        texts = enc.pop("texts")
        enc.pop("labels")
        enc = enc.to("cuda")
        enc["pixel_values"] = enc["pixel_values"].to(dtype)
        ids = model.generate(**enc, max_new_tokens=24, num_beams=1, do_sample=False)
        for pred, gold in zip(proc.batch_decode(ids, skip_special_tokens=True), texts):
            pr, pl = split_text(pred)
            gr, gl = split_text(gold)
            em += norm(pl) == norm(gl)
            tf += f1(pl, gl)
            ra += norm(pr) == norm(gr)
            n += 1
            if len(examples) < show:
                examples.append((gold, pred.strip()))
    ms = (time.perf_counter() - t0) * 1000 / max(n, 1)
    model.train()
    return {"n": n, "exact": round(em / n, 4), "token_f1": round(tf / n, 4), "role": round(ra / n, 4),
            "ms_per_crop": round(ms, 2), "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ax_labels")
    ap.add_argument("--model", default=str(CAPTION_DIR))
    ap.add_argument("--out", default=str(Path.home() / ".cache/rlca/ax-labeler"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--eval_n", type=int, default=3000, help="pairs per eval split (sampled, fixed seed)")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    dtype = torch.bfloat16
    proc, model = load(args.model, dtype)
    val, test = Pairs(args.data, "val", args.eval_n), Pairs(args.data, "test", args.eval_n)
    report = {"model": args.model, "before": {"val": evaluate(proc, model, val, dtype, show=12),
                                              "test": evaluate(proc, model, test, dtype)}}
    print(json.dumps(report["before"], indent=1), flush=True)
    if args.eval_only:
        return
    train = Pairs(args.data, "train")
    dl = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=collate(proc), num_workers=4, drop_last=True)
    steps = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    model.train()
    step, t0 = 0, time.perf_counter()
    while step < steps:
        for enc in dl:
            enc.pop("texts")
            enc = enc.to("cuda")
            enc["pixel_values"] = enc["pixel_values"].to(dtype)
            loss = model(**enc).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} {time.perf_counter() - t0:.0f}s", flush=True)
            if step % 1000 == 0 or step == steps:
                print(json.dumps({"step": step, "val": evaluate(proc, model, val, dtype)}), flush=True)
            if step >= steps:
                break
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    proc.save_pretrained(args.out)
    report["after"] = {"val": evaluate(proc, model, val, dtype, show=12), "test": evaluate(proc, model, test, dtype, show=12)}
    print(json.dumps(report["after"], indent=1), flush=True)
    json.dump(report, open(Path(args.out) / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
