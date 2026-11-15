"""Chooser-agnostic list quality: a frozen, off-the-shelf instruct LLM picks from each option list zero-shot.

Kev was trained on AX-style lists, so it scores lists by how AX-like they look. Here no model has seen any of our
list formats: the same records (the Kev records from relabel.py variants) are shown as a numbered list under the
user's command, and the model answers with a number (greedy, thinking off). Scored as in relabel_table.py: the
answer is right if it is any acceptable target, or "none" when the target isn't listed.

  python scripts/eval/frozen_pick.py --model Qwen/Qwen3-4B --records a/records.jsonl b/records.jsonl --out runs/pick
  -> <out>/<model>/<records path with / -> __>.json  ({"rows": [...], "summary": {...}})
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import torch

PROMPT = """{state}

On-screen elements:
{options}

Which element should be clicked to carry out the user's command? Reply with the element's number only. If none of the elements fits, reply {none}."""


def build(rec):
    q = rec["questions"]["element"]
    keys = [k for k in q["criteria"] if k != "none"] + (["none"] if "none" in q["criteria"] else [])
    lines = [f"{i}. {q['criteria'][k]}" for i, k in enumerate(keys) if k != "none"]
    none_i = len(lines)
    lines.append(f"{none_i}. none of these")
    return PROMPT.format(state=rec["state"], options="\n".join(lines), none=none_i), keys


def parse(text, keys):
    m = re.search(r"\d+", text)
    if not m:
        return None
    i = int(m.group())
    return keys[i] if i < len(keys) else None


@torch.inference_mode()
def run(model, tok, recs, batch):
    rows = []
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]["prompt"]))   # similar lengths per batch
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": recs[i]["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=6, do_sample=False)
        for i, ids in zip(idx, out[:, enc["input_ids"].shape[1]:]):
            ans = tok.decode(ids, skip_special_tokens=True)
            r = recs[i]
            pred = parse(ans, r["keys"])
            rows.append({"id": r["id"], "answer": ans, "pred": pred, "ok": pred in r["ok"], "listed": r["listed"],
                         "options": len(r["keys"]) - 1})
    return rows


def summary(rows):
    n = len(rows)
    listed = [r for r in rows if r["listed"]]
    return {"n": n, "acc": sum(r["ok"] for r in rows) / n, "in_list": len(listed) / n,
            "pick_when_listed": sum(r["ok"] for r in listed) / max(len(listed), 1),
            "unparsed": sum(r["pred"] is None for r in rows) / n,
            "options": statistics.median(r["options"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out", default="runs/pick")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--base", default="", help="strip this prefix from record paths when naming outputs")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    out_dir = Path(args.out) / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.records:
        name = str(Path(path).parent).removeprefix(args.base).strip("/").replace("/", "__") + ".json"
        if (out_dir / name).exists():
            print(f"skip {name}", flush=True)
            continue
        recs = []
        for rec in map(json.loads, open(path)):
            q = rec["questions"]["element"]
            prompt, keys = build(rec)
            recs.append({"id": rec["_meta"]["id"], "prompt": prompt, "keys": keys,
                         "ok": set(q.get("target") or [q["label"]]), "listed": q["label"] != "none"})
        t0 = time.time()
        rows = run(model, tok, recs, args.batch)
        s = summary(rows)
        s["minutes"] = round((time.time() - t0) / 60, 1)
        json.dump({"records": path, "model": args.model, "summary": s, "rows": rows}, open(out_dir / name, "w"))
        print(f"{name}: in list {s['in_list']:.1%}  pick|listed {s['pick_when_listed']:.1%}  acc {s['acc']:.1%}  "
              f"unparsed {s['unparsed']:.1%}  ({s['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
