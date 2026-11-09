# Research plan

**Working title:** Option lists from pixels: fast screen parsing that finds what accessibility misses.

**Claim to test:** a fast pixel parser (OCR + a GUI detector + icon labels) gives a computer-use agent a short,
well-labelled option list, so a small text model choosing from that list gets close to 2-8B screenshot models
at a fraction of the latency. Accessibility (AX) is an optional bonus, not the backbone.

Target venues: CVPR/ICCV/ECCV main track or NeurIPS (method + benchmark), or a CVPR 2027 GUI-agent / efficiency
workshop with the current results.

## Where things stand (from rl-computer-agent, see docs/background/option_results.md)

- Coverage is mostly solved. OCR fast (≤6 words) + GPA @640 covers about 85-91% of targets on held-out
  screens without AX, in about 50-60 ms on an M5.
- Labels are the bottleneck. On OSWorld-G the target is in the pixel list 79% of the time, but Kev picks it
  only about 36% of the time when it's there (ScreenSpot-Pro: about 32%). Florence-2 captions on 64×64 crops
  say too little on dense toolbars and brand icons.
- Lists are long: about 80-190 options per screen, near Kev's 255 cap.
- Ground truth leans toward AX. Labels were written from AX lists and the explorer only pressed AX elements,
  so "what AX misses" can't be measured yet.
- The sandbox capture is missing the toolbar strip on ~75% of screens and separate sheet/popup windows on ~14%.

## Contributions to build

1. **AX-gap detector.** Train or fine-tune the detector on AX-vs-pixel disagreements from the sandbox data
   (19,898 screens, 115 apps): AX gives free labels where it works; the model learns the gaps.
2. **Context-aware icon labels.** Replace the crop captioner with labels that use surrounding text, toolbar
   grouping and the icon's function. Main metric: chooser accuracy when the target is in the list.
3. **Ranking / pruning.** Cut the list to the options that matter without losing the target.
4. **Unbiased ground truth.** Hand-label every clickable element on a set of Mac screens (with AX recorded) so
   "AX misses X%" is measurable. Release it as a benchmark if possible.

## Experiments

- Coverage vs latency vs list length: pixel-only, AX-only, AX + pixels (ablation table).
- End-to-end grounding with a fixed small chooser on ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G, OmniACT,
  MMBench-GUI, and the hand-labelled Mac set.
- Baselines: OmniParser + GPT-4o (published), UI-Venus-1.5-2B/8B, GUI-Actor-2B, TinyClick, plain OmniParser v2.
- Ideally one multi-step task benchmark (OSWorld or a macOS one).

## Rules carried over

- Benchmarks stay out of training (perceptual-hash lookalike filter, see docs/background/benchmarks.md).
- GPA was likely trained on GroundCUA and MacPaw's YOLO on Screen2AX; don't report those as fair tests.
- Private explorer screenshots stay local; public sets and the sandbox data are fine to use.
