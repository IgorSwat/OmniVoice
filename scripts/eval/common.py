"""Shared plumbing for the eval scripts.

Kept in one place because every eval needs the same two awkward things: the
MPS loading dance, and a consistent reading of `data/test`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DTYPES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}

# `data/test` is out-of-distribution on purpose: real recordings, mp3, 44.1 kHz,
# unseen speakers, none of it in dataset.csv. Unlike the dev set it also exercises
# the *inference* topology -- a separate reference clip rather than the
# training-time same-utterance prefix -- which is what a clone model is judged on.
TEST_DIR = "data/test"


def read_test_inputs(test_dir=TEST_DIR):
    """Return (ref_texts, target_texts), both keyed by lowercase speaker name.

    `transcripts.txt` is `name | transcript`; `targets.txt` is `Name: text`.
    """
    refs = {}
    with open(os.path.join(test_dir, "transcripts.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                name, text = line.split("|", 1)
                refs[name.strip().lower()] = text.strip()

    targets = {}
    with open(os.path.join(test_dir, "targets.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                name, text = line.split(":", 1)
                targets[name.strip().lower()] = text.strip()

    missing = [s for s in refs
               if not os.path.exists(os.path.join(test_dir, f"{s}.mp3"))]
    if missing:
        raise SystemExit(f"missing reference audio in {test_dir}: {missing}")
    return refs, targets


def load_model(model_path, device="mps", dtype="fp16"):
    """Load OmniVoice for inference, keeping the codec off MPS.

    Two constraints force this shape: the Higgs codec cannot run on MPS at all
    (output channels > 65536), and loading straight onto MPS segfaults during
    weight loading. So: load on CPU, detach the codec, move only the LM.
    """
    import torch
    from omnivoice.models.omnivoice import OmniVoice

    torch_dtype = getattr(torch, DTYPES[dtype])
    model = OmniVoice.from_pretrained(model_path, device_map="cpu", dtype=torch.float32)
    if device == "mps":
        codec, fe = model.audio_tokenizer, model.feature_extractor
        model.audio_tokenizer = None
        model.to(device, torch_dtype)
        model.audio_tokenizer, model.feature_extractor = codec, fe
    else:
        model.to(device, torch_dtype)
    return model.eval()


def model_tag(model_path):
    """Short, filesystem-safe name for a checkpoint, for output directories."""
    p = model_path.rstrip("/")
    if p == "k2-fsa/OmniVoice":
        return "teacher"
    base = os.path.basename(p)
    # runs/foo/step_1000 -> foo_step_1000, so sibling checkpoints stay distinct
    parent = os.path.basename(os.path.dirname(p))
    if base.startswith("step_") and parent:
        return f"{parent}_{base}"
    return base or "model"
