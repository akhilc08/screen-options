"""Convert OmniParser-v2's icon caption weights (old remote-code Florence-2 layout) to native transformers.

Uses the rename rules from transformers' own Florence-2 conversion script (vendored in
.vendor/florence2) on top of the native florence-community/Florence-2-base-ft config and processor.
Writes ~/.cache/rlca/omniparser-caption.
"""
import glob
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, Florence2ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".vendor/florence2"))
import convert_florence2 as conv  # noqa: E402

BASE = "florence-community/Florence-2-base-ft"
OUT = Path.home() / ".cache/rlca/omniparser-caption"


def main():
    src = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--microsoft--OmniParser-v2.0/snapshots/*/icon_caption/model.safetensors"))[0]
    omni = load_file(src)
    model = Florence2ForConditionalGeneration.from_pretrained(BASE)
    cfg = model.config.vision_config
    pairs = []
    for s in range(len(cfg.embed_dim)):
        pairs += conv.vision_conv_embeddings(s)
        for b in range(cfg.depths[s]):
            pairs += conv.vision_spatial_block(s, b) + conv.vision_channel_block(s, b)
    pairs += conv.multi_modal_projector() + conv.language_model(omni)
    target = model.state_dict()
    new = {}
    for old, name in pairs:
        if old not in omni:
            continue
        w = omni[old]
        if old == "image_projection":
            w = w.t()
        if w.shape != target[name].shape:       # vocab rows padded for the added <image> token
            pad = torch.zeros_like(target[name])
            pad[: w.shape[0]] = w
            w = pad
        new[name] = w.contiguous()
    new["model.language_model.shared.weight"] = new["model.language_model.encoder.embed_tokens.weight"] = \
        new["model.language_model.decoder.embed_tokens.weight"] = new["lm_head.weight"] = \
        new.get("model.language_model.shared.weight", new["lm_head.weight"])
    missing, unexpected = model.load_state_dict(new, strict=False)
    print("missing:", missing, "\nunexpected:", unexpected)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    AutoProcessor.from_pretrained(BASE).save_pretrained(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
