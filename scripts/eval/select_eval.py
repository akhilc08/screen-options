"""Does a better option list help the model that picks from it? A fixed off-the-shelf selector on each list.

The selector (Qwen2.5-VL-7B-Instruct, unchanged) sees the screenshot, the instruction and a numbered list of
options (text elements: their OCR string; icons: OmniParser's caption, as the pipeline would give them; each with
its box in the image it sees) and answers with one option number. It is right if the option is a target.

Lists, all over the same candidates (data/cands/<set>/axgpa_omni.jsonl):
  ours@K   our instruction-aware ranker's top K (sim from <name>.inst<tag>.jsonl), K in --ks
  conf@K   detector confidence order, top K (task-agnostic ranking)
  full     every candidate, unranked (what a pixel parser hands over without ranking)
Icon captions are computed once per screen for all icons and cached (data/cands/<set>/<name>.captions.jsonl).

  python scripts/eval/select_eval.py --sets osworld_g screenspot_pro screenspot_v2 --inst_tag _384
  -> runs/select/<set>/rows.jsonl (resumable), runs/select/summary.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
SELECTOR = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT = ("You are operating a computer. Task: {inst}\n\nThe screenshot shows the screen. These are the on-screen "
          "elements you can act on, as [number] label @ (x1, y1, x2, y2) in the screenshot's pixels:\n{opts}\n\n"
          "Which element should be clicked to do the task? Answer with its number only.")


def captions(s, name, cands, cap):
    """{image: {element index: caption}} for every icon, cached."""
    path = DATA / "cands" / s / f"{name}.captions.jsonl"
    done = {r["image"]: {int(k): v for k, v in r["captions"].items()} for r in map(json.loads, open(path))} \
        if path.exists() else {}
    with open(path, "a") as f:
        for i, c in enumerate(cands):
            if c["image"] in done:
                continue
            img = Image.open(DATA / "public" / c["image"]).convert("RGB")
            idx = [k for k, e in enumerate(c["elements"]) if e["role"] != "text"]
            boxes = [(e["box"][0], e["box"][1], e["box"][0] + max(e["box"][2], 1), e["box"][1] + max(e["box"][3], 1))
                     for e in (c["elements"][k] for k in idx)]
            done[c["image"]] = dict(zip(idx, cap.captions(img, boxes) if boxes else []))
            f.write(json.dumps({"image": c["image"], "captions": done[c["image"]]}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{s} captions {i}/{len(cands)}", flush=True)
    return done


def fit(img, max_px):
    """Resize to at most max_px pixels, sides multiples of 28 (Qwen2.5-VL's patch grid), so the processor won't."""
    r = min(1.0, (max_px / (img.width * img.height)) ** 0.5)
    w, h = max(28, round(img.width * r / 28) * 28), max(28, round(img.height * r / 28) * 28)
    return img.resize((w, h)), w / img.width, h / img.height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["osworld_g", "screenspot_pro", "screenspot_v2"])
    ap.add_argument("--name", default="axgpa_omni")
    ap.add_argument("--inst_tag", default="_384")
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 25, 50])
    ap.add_argument("--conf_ks", nargs="+", type=int, default=[10, 50])
    ap.add_argument("--max_px", type=int, default=1280 * 28 * 28)
    ap.add_argument("--limit", type=int, default=500, help="examples per set")
    args = ap.parse_args()
    from screen_options.detect import Captioner
    from transformers import AutoModelForImageTextToText, AutoProcessor
    cap = Captioner()
    proc = AutoProcessor.from_pretrained(SELECTOR, min_pixels=28 * 28, max_pixels=args.max_px)
    model = AutoModelForImageTextToText.from_pretrained(SELECTOR, dtype=torch.bfloat16, device_map="cuda",
                                                        attn_implementation="sdpa").eval()
    summary = {}
    for s in args.sets:
        d = DATA / "cands" / s
        cands = list(map(json.loads, open(d / f"{args.name}.jsonl")))
        inst = {c["image"]: c["examples"] for c in map(json.loads, open(d / f"{args.name}.inst{args.inst_tag}.jsonl"))}
        caps = captions(s, args.name, cands, cap)
        out = ROOT / "runs/select" / s
        out.mkdir(parents=True, exist_ok=True)
        rows_path = out / "rows.jsonl"
        done = {(r["id"], r["list"]) for r in map(json.loads, open(rows_path))} if rows_path.exists() else set()
        n_ex = 0
        with open(rows_path, "a") as f:
            for c in cands:
                if n_ex >= args.limit:
                    break
                img = None
                els = c["elements"]
                labels = [e["text"] if e["role"] == "text" else caps[c["image"]].get(k, "icon")
                          for k, e in enumerate(els)]
                for ex in c["examples"]:
                    if n_ex >= args.limit:
                        break
                    sim = inst.get(c["image"], {}).get(ex["id"], {}).get("sim")
                    if sim is None:
                        continue
                    n_ex += 1
                    targets = {int(t[1:]) for t in ex["target_ids"]}
                    ours = sorted(range(len(els)), key=lambda i: -sim[i])
                    conf = sorted(range(len(els)), key=lambda i: -els[i].get("conf", 0.0))
                    lists = {f"ours@{k}": ours[:k] for k in args.ks}
                    lists.update({f"conf@{k}": conf[:k] for k in args.conf_ks})
                    lists["full"] = list(range(len(els)))
                    for lname, order in lists.items():
                        if (ex["id"], lname) in done:
                            continue
                        if img is None:
                            img, sx, sy = fit(Image.open(DATA / "public" / c["image"]).convert("RGB"), args.max_px)
                        gen, oom = None, False
                        for width in (60, 25):       # a very long list that does not fit: retry with shorter labels
                            opts = "\n".join(
                                f"[{j + 1}] {labels[i][:width]} @ ({round(els[i]['box'][0] * sx)}, {round(els[i]['box'][1] * sy)}, "
                                f"{round((els[i]['box'][0] + els[i]['box'][2]) * sx)}, {round((els[i]['box'][1] + els[i]['box'][3]) * sy)})"
                                for j, i in enumerate(order))
                            msgs = [{"role": "user", "content": [{"type": "image"},
                                                                 {"type": "text", "text": PROMPT.format(inst=ex["instruction"], opts=opts)}]}]
                            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                            enc = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            try:
                                with torch.inference_mode():
                                    gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
                                break
                            except torch.OutOfMemoryError:
                                del enc
                                torch.cuda.empty_cache()
                                enc = None
                        oom = gen is None
                        torch.cuda.synchronize()
                        ms = (time.perf_counter() - t0) * 1000
                        if oom:
                            f.write(json.dumps({"id": ex["id"], "list": lname, "n": len(order),
                                                "covered": bool(targets & set(order)), "pick": None, "correct": False,
                                                "answer": "", "ms": None, "tokens": None, "oom": True}) + "\n")
                            f.flush()
                            continue
                        ans = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                        m = re.search(r"\d+", ans)
                        pick = order[int(m.group()) - 1] if m and 1 <= int(m.group()) <= len(order) else None
                        f.write(json.dumps({"id": ex["id"], "list": lname, "n": len(order),
                                            "covered": bool(targets & set(order)), "pick": pick,
                                            "correct": pick in targets, "answer": ans, "ms": round(ms),
                                            "tokens": int(enc["input_ids"].shape[1])}) + "\n")
                        f.flush()
                if n_ex % 50 == 0:
                    print(f"{s} {n_ex}/{args.limit}", flush=True)
        rows = list(map(json.loads, open(rows_path)))
        summary[s] = {}
        for lname in sorted({r["list"] for r in rows}):
            rs = [r for r in rows if r["list"] == lname]
            cov = [r for r in rs if r["covered"]]
            ok = [r for r in rs if not r.get("oom")]
            summary[s][lname] = {"n": len(rs), "n_too_long": len(rs) - len(ok),
                                 "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4),
                                 "accuracy_fitting": round(sum(r["correct"] for r in ok) / max(len(ok), 1), 4),
                                 "coverage": round(len(cov) / len(rs), 4),
                                 "acc_if_covered": round(sum(r["correct"] for r in cov) / max(len(cov), 1), 4),
                                 "median_ms": sorted(r["ms"] for r in ok)[len(ok) // 2] if ok else None,
                                 "median_tokens": sorted(r["tokens"] for r in ok)[len(ok) // 2] if ok else None}
            print(s, lname, summary[s][lname], flush=True)
    json.dump(summary, open(ROOT / "runs/select/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
