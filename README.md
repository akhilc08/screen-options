# screen-options

Fast option lists from pixels for computer-use agents: OCR + a GUI detector (+ accessibility when it exists),
aimed at finding what accessibility misses in as little time as possible. Split out of
[rl-computer-agent](../rl-computer-agent) for a vision paper.

- Research plan: [docs/plan.md](docs/plan.md)
- Results so far: [docs/background/option_results.md](docs/background/option_results.md)
- Related work and benchmark rules: [docs/background](docs/background)

## Layout

| Path | What |
|---|---|
| `src/screen_options/` | `ocr.py` (Apple Vision), `detect.py` (GPA / OmniParser v2 + Florence-2 captions), `sources.py` (AX + OCR + detector merge), `ax.py`, `screen.py`, `explore.py` (macOS AX walk, capture, input) |
| `scripts/eval/` | Option-source benchmarks: `optbench_*` (v1, Arc + public sets), `optbench2_*` (GroundCUA, Screen2AX, sandbox), `replay_sources.py`, `overlay_sample.py`, `compare_sources.py` |
| `scripts/data/` | Public-set download and conversion, sandbox dataset build and labelling |
| `scripts/sandbox/` | EC2 Mac collector and setup |
| `scripts/setup/` | Florence-2 caption model conversion |
| `tests/` | Unit tests (`python -m pytest tests`) |

## Data

`data/` is git-ignored. Scripts expect the same layout as rl-computer-agent (`data/option_bench`,
`data/option_bench2`, `data/public`, `data/converted`, `data/sandbox_ds`). To reuse what's already there:

    ln -s ../rl-computer-agent/data data

Model caches are shared too (`~/.cache/rlca/coreml`, `~/.cache/rlca/omniparser-caption`).

## Setup

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

macOS sources (AX, Vision OCR, capture) need pyobjc plus Accessibility and Screen Recording permission.
