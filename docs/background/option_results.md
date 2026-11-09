# Option-source results (copied from rl-computer-agent docs/results.md, 2026-11-09)

Paths like `data/...` and `scripts/...` refer to rl-computer-agent unless the file also exists here.

## Where Kev's options come from: AX vs OCR vs detector (2026-10-25)

This is a replay of explorer run `data/explore/20261025-124135`: 712 unique Arc screenshots from 9 sites, run with `scripts/eval/replay_sources.py`. Raw output is in that run's `sources.jsonl` and `sources_summary.json` (see [data.md](data.md) for how to get the data). The Mac is an Apple Silicon laptop, and the detector and captioner run on MPS.

Each screen's options come from three sources:
- **Accessibility (AX):** the window's AX tree.
- **OCR:** Apple Vision text lines that don't overlap an AX element.
- **Detector icons:** OmniParser v2 YOLO boxes that overlap neither AX nor OCR.

Counts only include the page area, so the Arc sidebar is left out.

### Options found per screen

| Site | Screens | AX | + OCR | + detector icons |
|---|---|---|---|---|
| GitHub | 201 | 43 | +18 | +8 |
| Google | 153 | 33 | +13 | +8 |
| Amazon | 100 | 57 | +26 | +9 |
| Gmail | 100 | 68 | +10 | +18 |
| NeetCode | 89 | 39 | +11 | +15 |
| Docs | 43 | 16 | +12 | +6 |
| Canvas | 13 | 34 | +20 | +12 |
| Google Calendar | 8 | 72 | +54 | +33 |
| claude.ai | 5 | 1 | 0 | 0 |
| **All** | **712** | **44** | **+16** | **+10** |

The "+" columns are upper bounds. Some OCR lines are plain paragraphs and some detector boxes are decoration. A by-eye check of a sample is still to do.

### Latency per screen (median ms)

| Site | AX | OCR fast | OCR accurate | Detector |
|---|---|---|---|---|
| GitHub | 109 | 68 | 168 | 268 |
| Google | 108 | 66 | 170 | 268 |
| Amazon | 138 | 80 | 185 | 271 |
| Gmail | 111 | 66 | 171 | 266 |
| NeetCode | 124 | 54 | 153 | 271 |
| Docs | 102 | 43 | 125 | 266 |
| Canvas | 84 | 48 | 148 | 266 |
| Google Calendar | 87 | 56 | 160 | 274 |
| claude.ai | 75 | 35 | 107 | 262 |

Across all screens:

| Step | Median ms | p90 ms |
|---|---|---|
| Screenshot | 61 | 69 |
| AX | 111 | 190 |
| OCR fast | 61 | 111 |
| OCR accurate | 162 | 242 |
| Detector | 268 | 305 |
| Merge | 8 | 22 |
| Icon captions (Florence-2, 64×64 crops) | 203 | 312 |

Gmail's AX p90 is 463 ms. Kev 0.8B (4-bit MLX) takes about 170 ms per step.

### Cost before Kev runs (sources in parallel)

| Setup | Wait | Options per screen |
|---|---|---|
| AX only | ~110 ms | 44 |
| AX + fast OCR | ~110 ms (OCR finishes within the AX time) | up to 60 |
| AX + accurate OCR | ~170 ms | up to 60 |
| + detector | ~270 ms | up to 70 |

### By-eye check of the extras (2026-11-04)

These are hand judgments on 27 screenshots, 3 per site. `scripts/eval/overlay_sample.py draw` draws the AX, OCR-only and detector-only boxes; each extra was then marked as a real click target or not, and `score` tallies the marks. Judgments are in `overlays/judgments.jsonl` in the run folder.

| Site | AX | OCR extras | real | Detector extras | real |
|---|---|---|---|---|---|
| Amazon | 40 | 7.0 | 0.7 | 4.7 | 2.0 |
| Canvas | 33 | 12.0 | 4.7 | 10.3 | 1.7 |
| claude.ai | 1 | 0 | 0 | 0 | 0 |
| Docs | 15 | 8.0 | 2.0 | 4.7 | 3.7 |
| Google Calendar | 98 | 38.7 | 11.7 | 21.0 | 16.0 |
| GitHub | 26 | 16.0 | 0.3 | 4.0 | 1.0 |
| Gmail | 96 | 12.3 | 0 | 28.0 | 18.0 |
| Google | 17 | 2.3 | 0 | 0.3 | 0 |
| NeetCode | 35 | 10.3 | 3.7 | 22.0 | 4.0 |
| **All** | **40** | **11.9** | **2.6** | **10.6** | **5.1** |

- **Only about 1 in 5 OCR extras is a real target (2.6 of 11.9).** The rest are paragraphs, headings and labels. The real ones are mostly nav labels on pages where the AX page tree is missing.
- **About half the detector extras are real (5.1 of 10.6).** They're icon-only buttons AX doesn't expose: Gmail's star and importance markers, Docs/Sheets template tiles, theme toggles, carousel arrows, and mini-calendar dates.
- **No visible target was missed by all three sources** on these screens.
- **The extras matter most when Arc's AX tree has no web area.** That's 6 of 13 Canvas screens, 5 of 8 Calendar screens and 8 of 100 Amazon screens. There, AX only sees the browser chrome, and OCR plus the detector find the whole page. Next to try: setting `AXManualAccessibility` on Arc, which makes Chromium build the tree.
- **claude.ai isn't an AX problem.** All 12 visits captured `claude.ai/recents` still showing the loading spinner, so the page never finished loading in the background window.

So real targets per screen are roughly AX 40 + OCR 2.6 + detector 5.1, about 18% on top of AX, not the 60% upper bound. A short-line filter on OCR (nav labels, not sentences) would cut most of the OCR noise.

### Takeaways

- **AX + fast OCR in parallel is still the default.** It's free in wall time, but only about 6% more real targets, so OCR needs a short-line filter to keep the noise out of Kev's list.
- **The detector finds more real targets than OCR (5 vs 2.6 per screen) but costs 270 ms**, more than Kev itself. Run it when AX has no web area or the list looks thin, not on every screen.
- **Fixing the missing AX web area is the bigger win.** That's where most of the real extras come from.
- **Captions are weak on brand logos**, so icon text from the captioner can't be trusted alone.
- **claude.ai never finished loading in the background window.** The explorer needs to wait for it, or skip it.


## Option sources: coverage vs speed (2026-11-06)

Every option source scored on the same screens: 712 Arc screenshots from explorer run `20261025-124135` plus public screenshots (OmniACT 98 Mac screens, ScreenSpot-Pro 300, OSWorld-G 250). Scripts: `scripts/eval/optbench_build.py` (manifest, AX, Vision OCR), `optbench_detect.py` (pixel detectors), `optbench_score.py` (coverage). Outputs are in `data/option_bench/`.

A target is covered when some option's center falls inside one of its correct boxes. That is, clicking the option would hit it. The ground truths:

- **label:** labelled command targets on our Arc screens.
- **press:** elements whose press changed the screen.
- **public sets:** human answer boxes.

Label and press both lean toward AX, because labels were written from AX-based lists and the explorer only presses AX elements. The public sets have no AX at all, so they stand in for screens where AX fails.

Times are Mac medians (M5) per screen, with each source running alone. The Core ML columns run on all screens; the MPS rows are a 60-screen sample. UI-DETR was only run on the A10G.

### Single sources

| Source | Mac ms | Options/screen | label | press | OmniACT | ScreenSpot-Pro | SS-Pro icons | OSWorld-G |
|---|---|---|---|---|---|---|---|---|
| Blind grid, 150 cells (chance) | 0 | 148 | 48.7 | 44.2 | 45.6 | 18.0 | 12.7 | 27.5 |
| AX (recorded live) | 111 | 78 | 95.4 | 99.8 | | | | |
| Vision OCR fast | 66 | 62 | 73.8 | 72.6 | 58.8 | 24.0 | 3.4 | 31.6 |
| Vision OCR accurate | 169 | 62 | 75.8 | 71.1 | 63.5 | 23.0 | 3.4 | 39.0 |
| OmniParser v2 @1280, MPS (today's detector) | 132 | 124 | 84.5 | 81.4 | 82.3 | 83.0 | 87.3 | 75.9 |
| OmniParser v2 @1280, Core ML | 93 | 131 | 85.4 | 82.3 | 85.4 | 85.0 | 89.8 | 76.9 |
| **GPA-GUI-Detector @1280, Core ML** | **87** | 151 | 89.0 | 89.1 | 90.9 | **90.0** | 85.6 | 77.5 |
| **GPA-GUI-Detector @640, Core ML** | **32** | 116 | 87.3 | 87.1 | 89.6 | 57.7 | 53.4 | 79.0 |
| Screen2AX YOLO11l @1280, MPS | 157 | 152 | 82.2 | 84.7 | 89.4 | 78.0 | 77.1 | 73.7 |
| OmniParser icon_detect_v3 @1280, MPS | 574 | 147 | 90.1 | 87.1 | 90.9 | 64.0 | 61.0 | 77.3 |
| UI-DETR-1 @1600 (A10G only) | 336 (GPU) | 261 | 93.6 | 94.3 | 95.5 | 80.7 | 74.6 | 84.3 |

Detectors are shown at confidence 0.05. With the same budget of 100 boxes each on ScreenSpot-Pro, the scores are GPA 54, OmniParser v2 49, Screen2AX 45, v3 43, UI-DETR 41 and the blind grid 10. So GPA's lead isn't just from returning more boxes.

### Combinations

Sources run in parallel. When run together on the Mac, fast OCR plus GPA (Core ML) takes 60 ms at 640px and 99 ms at 1280px, both under the 111 ms AX walk.

| Combination | Mac ms (parallel) | Options/screen | label | label not from AX | press | OmniACT | ScreenSpot-Pro | OSWorld-G |
|---|---|---|---|---|---|---|---|---|
| AX + OCR fast | ~111 | 137 | 99.8 | 95.3 | 99.8 | | | |
| AX + OCR fast (≤6 words) + GPA @640 | ~111 | 187 | 100.0 | 99.7 | 99.8 | | | |
| OCR fast (≤6 words) + GPA @640, no AX | ~60 | 171 | 91.0 | 99.7 | 90.9 | 90.7 | 66.3 | 81.4 |
| OCR fast (≤6 words) + GPA @1280, no AX | ~99 | 204 | 92.2 | 99.7 | 92.4 | 92.1 | 91.3 | 80.0 |
| OCR fast (≤6 words) + OmniParser v2 @1280, no AX | ~94 | 185 | 92.6 | 99.0 | 92.8 | 92.5 | 89.3 | 81.8 |

### What this shows

- **Swap the detector for GPA-GUI-Detector via Core ML.** At 640px it takes 32 ms, against 132 ms for today's OmniParser v2 on MPS (268 ms in the earlier replay). It covers as much or more on every set. It's also MIT-licensed, while OmniParser v2's weights are AGPL. fp16 Core ML stays within 1–3 points of the GPU runs.
- **640px is enough for normal windows. Dense 4K screens need 1280px.** ScreenSpot-Pro coverage goes from 58% at 640px to 90% at 1280px, while every other set is about the same. Switch to 1280px only when the screen is large or crowded.
- **The pixel sources add no wait.** Fast OCR plus GPA finishes inside the AX walk, so AX + OCR + detector costs about the same ~111 ms as AX alone, and the merged list reaches 100% of labelled targets.
- **Without AX, OCR + GPA reaches about 91% on ordinary screens.** The misses are on dense pro apps (ScreenSpot-Pro at 640px) and Linux desktops (OSWorld-G, about 80%).
- **The cost is a longer list:** about 187 options per screen instead of 137. That makes Kev's prompt longer and pushes toward its 255-option cap. Kev also hasn't been trained on detector-only answers yet (see the caption bug in label prep). Pixel options should go in when AX is thin, or be ranked, rather than appended in full.
- **Not worth it:**
  - OmniParser icon_detect_v3: 4× slower than v2 on the Mac and worse on dense screens.
  - UI-DETR-1: the best raw coverage, but it returns 260–300 boxes and takes 336 ms even on a GPU.
  - Screen2AX's macOS-trained YOLO: slower than GPA and worse on every set.
- **A small model listing the elements doesn't work yet:**
  - Florence-2-base region proposals return about one box per screen, usually the whole page, so they cover almost nothing (0–3%).
  - ScreenVLM-316M (IBM's screen parser) writes out the whole page as tagged boxes. It hit its 4,096-token cap on every screen and took about 340 s per screen on an A10G. On the 6 screens it finished, it found 6 of 19 pressed elements. Even a 100× faster runtime would be slower than the whole AX + OCR + detector stack.

### Where each step ran

Nothing here touched live apps. AX was not walked again: the scores use the AX trees the explorer recorded live on the owner's Mac (Arc on 2026-10-25, `data/explore/20261025-124135`). Everything else ran on saved screenshots.

| Step | Ran on | Input |
|---|---|---|
| AX | owner's MacBook (M5), live, during the 2026-10-25 explorer run | recorded `elements` and `ms.ax_window` in `states.jsonl` |
| Vision OCR, fast and accurate | owner's MacBook | saved screenshots (`optbench_build.py`) |
| Detector coverage on all screens | AWS g5.xlarge (A10G) | the same screenshots, uploaded with the owner's OK and deleted afterward |
| Detector timing | owner's MacBook: Core ML on all screens, MPS on a 60-screen sample | saved screenshots (`optbench_detect.py`) |
| Core ML export | AWS box (needs `numpy<2.3` and torch ≤ 2.7; coremltools 9 fails on the Mac's torch 2.14) | Ultralytics `export(format="coreml", half=True)` |
| Scoring | owner's MacBook | `optbench_score.py` |

Not tested yet:
- **Native apps.** The about 440 explorer screenshots of Slack, Notion, Notes, Spotify and Zed in `data/native/` already have recorded AX. Other apps would need fresh look-only captures.
- **Independent ground truth.** Targets that every source missed are invisible to these labels.

### Running it on an AWS Mac

Every step except AX runs on saved files, so on an EC2 Mac instance:

1. Copy `data/option_bench/manifest.jsonl` and the screenshots it lists. Don't copy the private explore screenshots unless the owner signs off; the public sets alone are fine.
2. Run `optbench_build.py`, then `optbench_detect.py --source gpa_coreml:640` (and the others), then `optbench_score.py`.
3. Copy the Core ML packages from `~/.cache/rlca/coreml/`, or export them again in an environment with `numpy<2.3`.

Live AX needs a bit more:

- **Permissions.** The Python or terminal process needs Accessibility and Screen Recording access. On an EC2 Mac these are granted once through a GUI session (VNC / Screen Sharing), or through an MDM profile.
- **Apps.** They have to be installed, signed in where it matters, and open, with a real window on the virtual display.
- **Scripts.**
  - `scripts/eval/compare_sources.py` walks AX on the current window of each running app and times it.
  - `scripts/data/explore_native.py` records full runs in the layout `optbench_build.py` reads (`states.jsonl` + `shots/`).
- **Timings.** EC2 Macs are M1 / M2 / M2 Pro / M4, not the owner's M5, so expect different numbers. Compare sources on the same machine, not against this table.



## Option sources v2: 3,158 screens, 153 apps (2026-11-07)

Which detector and merge rule to use to build Kev's option list, scored against dense element boxes
(`scripts/eval/optbench2_*.py`; detectors on an A10G, Apple Vision OCR on an EC2 Mac). Recall = share of
clickable elements with an option center inside them. Options = list length after merging near-duplicates.

- **GroundCUA** (2,131 screens, 87 desktop apps, people boxed every element). GPA-GUI-Detector was almost
  certainly trained on it (mean best IoU 0.80 vs ~0.48 for the others; it boxes 97% of the non-clickable
  "Information Display" text), so GPA's numbers there are not a fair test.
- **Screen2AX** (158 macOS screens, MacPaw). MacPaw's YOLO was trained on this data, so its numbers are not fair here.
- **Sandbox** (869 screens from our EC2 Mac collector, ~50 apps and sites). Targets are the live AX elements, so it
  measures how much of AX a pixel method recovers. It cannot show what AX misses; that needs hand labels.

### Clickable recall at conf 0.15 (union)

| method | Screen2AX (Mac, held out) | Sandbox (AX recovered) | GroundCUA | options (S2AX) | ms |
|---|---|---|---|---|---|
| OCR fast (≤6 words) | 51.2 | 44.1 | 30.5 | 32 | 48 |
| GPA @640 | 81.1 | 50.4 | *97.2 (trained on it)* | 41 | 28 |
| OmniParser v2 @640 | 66.2 | 45.2 | 68.1 | 25 | 27 |
| OmniParser v3 @1280 | 67.9 | 52.3 | 78.9 | 27 | 207 |
| UI-DETR-1 @1600 | 91.0 | 64.3 | 93.7 | 105 | 320 |
| OCR + GPA @640 | 84.9 | 57.8 | *97.6* | 72 | 50 |
| OCR + UI-DETR @1600 | 91.8 | 64.8 | 94.0 | 131 | 320 |
| OCR + UI-DETR @1600, conf 0.3 | 87.7 | 62.4 | 88.3 | 83 | 320 |
| OCR + GPA @640, conf 0.05 | 86.7 | 59.6 | *98.8* | 82 | 50 |

### What this shows

- **GPA stays the pick.** At the same list length (~82 options) OCR + GPA scores 86.7 on held-out Mac screens vs
  87.7 for OCR + UI-DETR, at 50 ms vs 320 ms. UI-DETR's lead at conf 0.15 comes from listing 105-130 options, most
  of them noise (precision 39% vs 49-54% for GPA).
- **OCR is worth adding**: +3-4 points over a detector alone at little cost. Fast OCR is enough; accurate OCR is 5x
  slower (250 ms) and adds about 1 point when combined.
- **Two detectors don't help.** Every two-detector "fill" combination scores below OCR + GPA alone.
- **Pixels recover only 55-65% of AX on real Mac UI**, so AX remains the backbone wherever it exists. *Correction
  (2026-11-08): this understates pixels. The sandbox screenshots are missing the window toolbar on ~75% of
  screens and sheets/popups on ~14% (see "Sandbox dataset" below), so the detector never saw those AX targets.* Gating the
  detector to AX-poor screens (gate10) gives the shortest lists (81 vs 100 options) at equal AX recall, but whether
  the detector's extra boxes are real targets AX missed can only be settled with hand labels.
- Decision: **AX + fast OCR (≤6 words) + GPA @640** (union, conf 0.05-0.15). About 120-130 ms on an M5 with the
  parts in parallel.

## Sandbox dataset (2026-11-08)

Three EC2 Macs ran the collector for 20 h each (`scripts/sandbox/collect.py`): 33,290 screenshots. After
`scripts/data/sandbox_dataset.py` (clean, split, sources, visible):

- **19,898 screens, 115 apps/sites** (40 websites); 13,258 near-duplicates and 134 blank frames dropped. Light and
  dark mode, four window sizes. 1,481 screens are AX-poor (< 8 labelled AX elements).
- **Split by app, frozen** (`data/sandbox_ds/splits.json`): train 80 apps / 14,291 screens, val 12 / 2,082, test 23 / 3,525.
- **Option list per screen** (AX + fast OCR + GPA@640): on average 61 AX, 13 OCR-only and 7 detector-only options;
  OCR 24 ms and GPA 23 ms median on the laptop.
- **Capture gap.** The collector's window capture left the toolbar strip blank on 14,936 screens (75%; Safari's
  tab bar and toolbar, app toolbars) and missed sheets, popovers, autocomplete lists and open menus, which are
  separate windows, on ~2,800 screens. The AX tree still lists those controls, so AX targets stay valid, but
  each AX option now carries `visible` and screens with more than 30% hidden carry `capture_mismatch`. Worth
  checking whether Kev's runtime capture has the same gap on macOS 26.
- **Commands (Claude drafts)** on 200 screens (100 train, 100 test, 103 apps), from `scripts/data/sandbox_label.py`:
  2,674 commands and 893 icon captions; 20% of commands target a detector-only icon, 10% are absent. The test
  drafts still need a human pass.
