# Benchmarks and baselines

This is the evaluation plan for Kev, the "hands" of the agent. Given a command and the elements on screen, it picks the element and action, fast, and knows when to ask.

We evaluate it as a component, the way grounding papers (UI-Venus, GUI-Actor, Phi-Ground, TinyClick) do: one-step decisions, not full-task success. Results go in [results.md](results.md) and related work in [related_work.md](related_work.md).

## What we measure

| Metric | What it is |
|---|---|
| **Accuracy** | The picked element is one of the correct ones. For screenshot models, the click lands inside the target box. |
| **Time per decision** | Wall time per command, measured the same way for every model: on the AWS A10G for relative speed, and in a short run on the Mac (4-bit MLX, M5) for real speed |
| **Calibration** | Expected calibration error (ECE) of the model's confidence |
| **Act alone at 90% / 95%** | Share of commands the gate could run without asking while staying at least 90% or 95% correct |
| **Asking** | On commands labelled ambiguous: how often it asks. On commands whose target isn't on screen: how often it says "none". |
| **Option-list ceiling** | How often the right element is in the list at all, reported next to every list-based score so the comparison with screenshot models is fair |

## Benchmark set

| Benchmark | Input we give Kev | Why | Status |
|---|---|---|---|
| **Mind2Web** | Its own element lists (real HTML elements saved per step) | The standard benchmark for element-list models (MindAct and most web agents). Same task as ours, with cross-task, cross-website and cross-domain test splits. | To download and convert. Plan: train on its training split, as MindAct did, and report the three test splits. |
| **ScreenSpot-v2** | Our pixel parser: Apple Vision OCR + OmniParser icon detector + captions | The standard grounding benchmark. Run through a parser the same way OmniParser reports it. | Converted (1,272 examples, target in list 97%) |
| **ScreenSpot-Pro** | Pixel parser | The hard grounding benchmark: high-resolution professional apps | Converted (1,580, target in list 77%) |
| **OSWorld-G** | Pixel parser | Desktop grounding (Ubuntu), from OSWorld runs | Converted (510, target in list 79%) |
| **MMBench-GUI L2** | Pixel parser | Multi-platform grounding, including macOS | Annotations and images downloaded, not converted yet |
| **OmniACT** | Pixel parser | Native Mac apps with task-style commands. No published grounding scores, so it's our own comparison only. | Converted (7,068 commands on 99 screens). All of it is held out. |
| **Our Mac set** | Real macOS accessibility tree + OCR + detector | The setting Kev is built for. It's the new contribution. | The current labels contain private screens. A releasable version is still to build, from public sites and clean test accounts. |

Screenshot benchmarks run through our pixel parser are the **no-accessibility worst case**: every element has to come from OCR and icon detection. We report them as that. We don't generate fake accessibility trees from screenshots (Screen2AX style) for them, because a predicted tree is just another pixel parser. Kev's real advantage is the tree the app itself provides.

Optional later: OSWorld in its accessibility-tree mode, for full-task success with an existing planner and Kev as the executor (the way Jev-Mobile tests Jev).

## Keeping benchmarks out of training

- **OmniACT is held out completely.** Its train, val and test splits share the same 99 screens, so all 92 test screens appeared in training under the old split. Its old "58%" test score was inflated.
- **Lookalike filter.** `build_records.py` drops any training screenshot within 6 bits (out of 256) of a benchmark screenshot's perceptual hash. Of the about 2,500 OS-Atlas screens, 70 matched, covering about 900 examples, mostly Linux screens matching OSWorld-G and some ScreenSpot-v2. One ScreenSpot-v2 screen is pixel-identical to an OS-Atlas screen; the two datasets come from the same team.
- **Val is Mac labels only** (429 commands), so checkpoint choice never sees a benchmark.
- **Kev's pre-training has no screen data.** Its model card lists text classification, documents and code only. Qwen's pre-training is unknown, as it is for every open model.

Training data after these changes: about 23,850 examples, down from 28,155. That's 11,852 Mac (5,926 commands × 2), 6,000 OS-Atlas macOS and 6,000 OS-Atlas Linux. The OS-Atlas pools are capped. OmniACT was the only native-app training data with task-style commands, so native-app data has to come from the overnight explorer.

## Models we compare against

Every model in a group gets the same input and is scored the same way.

### 1. Same element list, different decision model

These are the main comparison: they answer whether Kev is the best decision model for this design.

| Model | Size | Notes |
|---|---|---|
| **Kev 0.8B** (ours, SFT) | 0.8B | |
| Kev 4B | 4B | Bigger. Kev's README reports much better act-alone coverage on text tasks (0.52 vs 0.14 at a 5% error budget). |
| **Jev** (TypeSafe, hosted) | Not disclosed | The closed model Kev is modelled on. Needs an API key. Run only on public or clean screens, since the data leaves the machine. |
| **MindAct** (DeBERTa-v3 ranker + Flan-T5 base/large/XL) | 0.09–3B | Published element-list computer-use models (Mind2Web). Web-trained. |
| **WebLINX** (Sheared-LLaMA 1.3B, Flan-T5) | 0.25–2.7B | Published element-list web agents |
| Small general LLMs, prompted: Qwen3 0.6B/1.7B, Qwen3.5 0.8B/2B, Gemma 3 270M/1B, Llama 3.2 1B, LFM2 700M, LFM2.5 1.2B | 0.27–2B | Same list, no GUI training. Shows what our training adds. |
| model2vec similarity | Tiny | Speed floor |
| A fast frontier model (e.g. Claude Haiku) | Large, hosted | Accuracy ceiling for the list approach |

Script: `scripts/eval/list_baselines.py`.

### 2. Screenshot models, on the same benchmarks

These show the other design's speed and accuracy, and are candidates for a fallback when Kev isn't confident.

| Model | Size | Notes |
|---|---|---|
| **UI-Venus-1.5-2B** | 2B | Strongest current 2B (reports 57.7% on ScreenSpot-Pro) |
| **GUI-Actor-2B** (Microsoft) | 2B | Pointer head over image patches. Needs its own environment (transformers 4.51). |
| **Qwen3-VL-2B** | 2B | Plain vision model, as a floor |
| **ShowUI-2B** | 2B | Small image budget, built for speed |
| **TinyClick** (Samsung) | 0.27B | The one screenshot model in our speed range (reports about 250 ms) |
| UI-Venus-1.5-8B | 8B | Accuracy ceiling at a reasonable size |

Script: `scripts/eval/vlm_grounding.py` (GPU box) and `scripts/eval/vlm_mac_speed.py` (short Mac timing). Each model uses its official prompt and coordinate format; sources are in the script.

### 3. Parser + large model

**OmniParser + GPT-4o**, from its published ScreenSpot-Pro number (39.6). That's the closest published setup to running Kev on screenshots through a parser.

## Rules for running

- Long runs go on the AWS GPU box. The Mac only gets short timing runs (about 25 commands per model), with the owner's OK.
- Benchmarks are never used for training. The lookalike filter runs on every record build.
