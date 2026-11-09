# Related work

The closest prior work to Kev as the agent's hands: a small, fast model that picks an element from a text list of what's on screen, on-device, with calibrated confidence for an act / ask / confirm gate. Collected 2026-11-05. Numbers are quoted from each paper.

**Main finding:** no paper matches the whole design: a sub-1B text model picking from an accessibility list on a laptop, in about 0.17 s per decision, with calibrated asking. The nearest are Jev-Mobile and V-Droid (text candidates plus a fast chooser, but served remotely or on GPUs), MindAct (small ranker plus selector, no timing), and TinyClick / Phi-Ground (screenshot grounding in about 0.12–0.25 s, hardware not named). On-device numbers for 2–8B GUI agents on phones are about 7–40 s per step.

## Same design: text element list + a fast chooser

| Paper | Model | Input | Speed reported | Accuracy | Difference from us |
|---|---|---|---|---|---|
| [Jev-Mobile](https://arxiv.org/abs/2609.30186) (2026, Android) | Jev (hosted, size undisclosed) as executor under a cloud planner | Accessibility-tree candidates | No per-step number. Executor time per successful task 5.2 s vs 12.4 s (UI-TARS-7B). Total time per success −32.7%, cost −73.4%. | AndroidWorld 79% (step-wise VLM 84%) | Hosted, phone, no calibration study. Nearly the same split as ours. |
| [V-Droid](https://arxiv.org/abs/2503.15937) (MobiCom '26) | Llama-3.1-8B 4-bit + MLP scoring head | Accessibility XML | 0.74 s per decision on 2× RTX 4090. About 2.5 s estimated on a phone NPU. | AndroidWorld 59.5% | Closest mechanism (a head scores every candidate). 10× our size, on GPUs. |
| [Mind2Web / MindAct](https://arxiv.org/abs/2306.06070) (NeurIPS '23) | DeBERTa-v3-base ranker → Flan-T5 (0.2–3B) selector | HTML elements | None | Element accuracy 55.1 (Flan-T5-XL, cross-task) | The classic version of our design, web only |
| [AutoDroid](https://arxiv.org/abs/2308.15272) (MobiCom '24) | Vicuna-7B on the phone | Simplified UI tree | 22.7–39.7 s per step on a Snapdragon 8 Gen 2. 4.2–8.8 s on an RTX 3090. | 57.7% action accuracy | Same input on-device, about 100× slower |
| [CORE](https://arxiv.org/abs/2510.15455) (NeurIPS '25) | Local 7–9B ranks UI blocks, GPT-4o picks | XML UI | 61 s per task on an RTX 4090D. 170 s on a phone. | DroidTask 69.2% | Shows 7B on a phone is far too slow |

## Small screenshot models that report speed

| Paper | Model | Speed reported | Accuracy |
|---|---|---|---|
| [TinyClick](https://arxiv.org/abs/2410.11871) (Samsung) | 0.27B (Florence-2), for on-device voice assistants | About 250 ms (hardware not given) | ScreenSpot 73.8%, OmniACT 58.3% |
| [Phi-Ground](https://arxiv.org/abs/2507.23779) (Microsoft) | 4B / 7B | 122 ms (4B) on a GPU (not named) | ScreenSpot-v2 92.3%, Pro 51.5% |
| [Auto-GUI](https://arxiv.org/abs/2309.11436) | 0.2B / 0.7B | 0.19 / 0.59 s per action | AITW 74.3% |
| [PastForward](https://arxiv.org/abs/2609.32166) (2026) | Qwen3-VL-2B on phones / Jetson | 6.7–10.9 s per step after their speedups | About 17–22% task success |
| [Octopus v2](https://arxiv.org/abs/2404.01744) | 2B, Android API calls (not GUI) | 0.36 s (A100). 1.1–1.7 s on a phone. | About 99% function-call accuracy |

## Small on-device agents that report no speed

[Ferret-UI Lite](https://arxiv.org/abs/2509.26539) (Apple, 3B; ScreenSpot-Pro 53.3), [MAI-UI](https://arxiv.org/abs/2512.22047) (2B on-device + 32B cloud), [GUI-Actor](https://arxiv.org/abs/2506.03143) (2B; pointer head over image patches), and [ShowUI](https://arxiv.org/abs/2411.17465) (training speedup only). The survey [Efficient GUI Agents](https://arxiv.org/abs/2609.02309) (2026) finds most papers don't report wall-clock latency.

## Accessibility list vs. screenshots

- [OSWorld](https://arxiv.org/abs/2404.07972): for 2024 models, the accessibility tree roughly doubled task success over screenshots. Set-of-Mark was erratic across models.
- [UFO2](https://arxiv.org/abs/2504.14603): accessibility-first with vision fill-in beat either alone (o1 on WAA: UIA 25.3, vision 20.8, hybrid 27.9%).
- [Windows Agent Arena](https://arxiv.org/abs/2409.08264): merging UIA with OmniParser helps strong models. Weak models gain little or get worse.
- [Agent S2](https://arxiv.org/abs/2504.00906), [UGround](https://arxiv.org/abs/2410.05243): top agents moved to pure screenshots, with large models.
- [Screen2AX](https://arxiv.org/abs/2507.16704): **only 33% of macOS apps offer full accessibility support.** That's the main risk for an accessibility-first Mac agent.
- [OmniParser V2](https://huggingface.co/microsoft/OmniParser-v2.0): parser + GPT-4o gets 39.6 on ScreenSpot-Pro, below today's 2B grounders. About 0.6 s per frame on an A100.
- Abstention for screenshot models needs extra machinery: [SafeGround](https://arxiv.org/abs/2602.02419) (sampling) and [HyperClick](https://arxiv.org/abs/2510.27266) (trained confidence). Kev's pointer head gives a distribution including "none" in one pass.

## Jev and computer use

Jev is TypeSafe's hosted, closed "System One" decision model ([docs](https://docs.typesafe.ai/api)). Kev is its open, small counterpart with the same API. TypeSafe has no official computer-use product. Third-party projects exist, all tested on tiny samples (3–6 tasks):
- [jev-macos-loop](https://github.com/jcpsimmons/jev-macos-loop): ScreenCaptureKit + OmniParser + Apple Vision OCR + AX, with Jev picking element ids. About 300 ms per decision, about 380 ms per cycle. Nearly our stack, with a hosted model.
- [jev-use](https://github.com/savka777/jev-use): voice-driven macOS on the AX tree only. Asks before low-confidence or destructive picks.
- [typesafe-computer-use](https://github.com/awlevin/typesafe-computer-use): macOS OCR + AX, Jev choosing from up to 255 options with a confidence gate. Claims about 1.5 s per step vs about 5.5 s for a screenshot-based frontier model.
- [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (browser-use): a numbered DOM table, no screenshots.

No project or paper uses Kev for computer use.

## Where we fit

What's new here:
- **Speed:** sub-second decisions on a laptop with a sub-1B model. We measured screenshot models 10–50× slower on the same Mac (see [results.md](results.md)).
- **Calibrated asking:** confidence over an element list, driving act / ask / confirm, evaluated as such.
- **Desktop, not phone:** Mac screens have hundreds of elements, menu bars, multiple windows and patchy accessibility.
- **Open and on-device,** unlike Jev-Mobile and the Jev projects.

To earn a "better than Jev" claim, Jev has to run on the same Mac benchmark with the same lists.

**Papers to position against:** Jev-Mobile, V-Droid, MindAct, TinyClick, OmniParser.
