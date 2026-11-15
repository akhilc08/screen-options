"""Coverage@K curves for the paper (leakage-filtered): ours, zero-shot SigLIP, detector confidence, UI-Venus-1.5-8B's
click as a ranking, and the 'listed' ceiling. -> docs/cvpr27/paper/fig/cov_k.pdf"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SETS = [("osworld_g", "OSWorld-G"), ("screenspot_pro", "ScreenSpot-Pro"), ("screenspot_v2", "ScreenSpot-v2")]
KS = [1, 5, 10, 25, 50, 100]
main = json.load(open(ROOT / "runs/rank_omni_384_clean.json"))
zs = json.load(open(ROOT / "runs/rank_zs384c_clean.json"))
gr = json.load(open(ROOT / "runs/grounder_combo_384_clean.json"))
series = [("Ours", "#2a78d6", "o", "-"), ("UI-Venus-1.5-8B click", "#eb6834", "s", "--"),
          ("Zero-shot SigLIP", "#1baf7a", "^", "-."), ("Detector confidence", "#eda100", "D", ":")]
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.95), sharey=False)
for ax, (s, title) in zip(axes, SETS):
    cur = {"Ours": [main[s]["sim"][f"cov@{k}"] for k in KS],
           "Zero-shot SigLIP": [zs[s]["sim"][f"cov@{k}"] for k in KS],
           "Detector confidence": [main[s]["conf"][f"cov@{k}"] for k in KS]}
    v = gr[s]["venus-8b"]                       # K = 1, 3, 5, 10, 25, 50
    gk, gv = [1, 5, 10, 25, 50], [v[0], v[2], v[3], v[4], v[5]]
    for name, col, mk, ls in series:
        xs, ys = (gk, gv) if name.startswith("UI-Venus") else (KS, cur[name])
        ax.plot(xs, [100 * y for y in ys], color=col, marker=mk, ms=3.2, lw=1.4, ls=ls, label=name)
    ax.axhline(100 * main[s]["sim"]["full"], color="#888", lw=0.8, ls=(0, (2, 2)))
    ax.text(1.05, 100 * main[s]["sim"]["full"] + 1.2, "on the list (ceiling)", color="#555", fontsize=6.5)
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.minorticks_off()
    ax.set_title(title, fontsize=8); ax.set_xlabel("K (list length)")
    ax.set_ylim(0, 103); ax.grid(axis="y", lw=0.4, color="#ddd"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Coverage@K (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
out = ROOT / "docs/cvpr27/paper/fig"
out.mkdir(exist_ok=True)
fig.savefig(out / "cov_k.pdf", bbox_inches="tight"); fig.savefig(out / "cov_k.png", dpi=160, bbox_inches="tight")
