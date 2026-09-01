#!/usr/bin/env python3
"""Generate listenable samples from `data/test` for a checkpoint.

The held-out probe used throughout the distillation work: nine real recordings
(podcast, conference talk, product demo, interview), none of them in
`dataset.csv`, cloned with a separate reference clip rather than the
training-time same-utterance prefix. Output lands in `./tmp/<tag>/` ready to
listen to.

Every generation is seeded per cell, so the same `--seed` reproduces the same
audio, and two checkpoints run at the same seed differ only by their weights ---
which is what makes A/B comparison meaningful rather than a sampler lottery.

    # teacher, each speaker reading their own line
    python scripts/eval/generate_samples.py

    # a trained checkpoint at 16 steps, side by side in ./tmp
    python scripts/eval/generate_samples.py --model runs/prefix_blocked --num-step 16

    # every voice x every line (81 clips) -- the WER-sweep layout
    python scripts/eval/generate_samples.py --cross

Pair two runs into a blind listening test with `--blind`:

    python scripts/eval/generate_samples.py --blind tmp/teacher tmp/prefix_blocked
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DTYPES, TEST_DIR, load_model, model_tag, read_test_inputs  # noqa: E402


def make_blind(dirs, out_dir, seed):
    """Copy two result directories into randomised A/B pairs for blind listening."""
    a_dir, b_dir = dirs
    names = sorted(set(os.listdir(a_dir)) & set(os.listdir(b_dir)))
    names = [n for n in names if n.endswith(".wav")]
    if not names:
        raise SystemExit(f"no common .wav files between {a_dir} and {b_dir}")

    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    key = {}
    for n in names:
        stem = n[:-4]
        letters = ["A", "B"]
        rng.shuffle(letters)
        for letter, src_dir in zip(letters, [a_dir, b_dir]):
            dst = f"{stem}_{letter}.wav"
            shutil.copy(os.path.join(src_dir, n), os.path.join(out_dir, dst))
            key[dst] = src_dir

    sheet = os.path.join(out_dir, "listening_sheet.md")
    with open(sheet, "w", encoding="utf-8") as f:
        f.write("# Blind A/B\n\nSame reference, same text, same seed within each "
                "pair. A/B randomised per row.\n\n"
                "- **prefer** — A, B, or `=` if you cannot tell\n"
                "- **conf** — 1 (guessing) / 2 (leaning) / 3 (obvious)\n\n"
                "| pair | prefer | conf | what differed |\n|---|---|---|---|\n")
        for n in names:
            f.write(f"| {n[:-4]:<12s} |  |  |  |\n")

    # The key must not sit next to the audio, or it is one `ls` from spoiling the test.
    key_path = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".",
                            f".{os.path.basename(out_dir.rstrip('/'))}_key.json")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "key": key}, f, indent=2)
    print(f"{len(names)} pairs -> {out_dir}\nsheet: {sheet}\nkey:   {key_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="k2-fsa/OmniVoice",
                    help="HF id or local checkpoint dir (default: the teacher)")
    ap.add_argument("--test-dir", default=TEST_DIR)
    ap.add_argument("--out-dir", default="tmp")
    ap.add_argument("--tag", default=None,
                    help="subdirectory name (default: derived from --model)")
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--t-shift", type=float, default=0.1)
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--cross", action="store_true",
                    help="every speaker x every target (81) instead of the diagonal (9)")
    ap.add_argument("--speakers", nargs="*", default=None, help="subset of voices")
    ap.add_argument("--targets", nargs="*", default=None, help="subset of target texts")
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate files that already exist")
    ap.add_argument("--blind", nargs=2, metavar=("DIR_A", "DIR_B"), default=None,
                    help="skip generation; pair two existing result dirs blind")
    args = ap.parse_args()

    if args.blind:
        make_blind(args.blind, os.path.join(args.out_dir, "blind"), args.seed)
        return

    import torch
    import soundfile as sf
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    refs, targets = read_test_inputs(args.test_dir)
    speakers = sorted(args.speakers or refs)
    target_names = sorted(args.targets or targets)
    for s in speakers:
        if s not in refs:
            raise SystemExit(f"unknown speaker {s!r}; have {sorted(refs)}")
    for t in target_names:
        if t not in targets:
            raise SystemExit(f"unknown target {t!r}; have {sorted(targets)}")

    if args.cross:
        cells = [(s, t) for s in speakers for t in target_names]
    else:
        # Diagonal: each voice reads the line written for it.
        cells = [(s, s) for s in speakers if s in targets]

    tag = args.tag or model_tag(args.model)
    out = os.path.join(args.out_dir, tag)
    os.makedirs(out, exist_ok=True)

    def path_for(sp, tg):
        stem = sp if sp == tg and not args.cross else f"{sp}__{tg}"
        return os.path.join(out, f"{stem}.wav")

    todo = [(s, t) for s, t in cells
            if args.overwrite or not os.path.exists(path_for(s, t))]
    if not todo:
        print(f"nothing to do — {len(cells)} files already in {out}/")
        return

    print(f"model {args.model}  ->  {out}/")
    print(f"{len(todo)} of {len(cells)} clips to generate "
          f"(num_step={args.num_step}, t_shift={args.t_shift}, "
          f"guidance_scale={args.guidance_scale})")

    t0 = time.time()
    model = load_model(args.model, args.device, args.dtype)
    print(f"loaded in {time.time()-t0:.1f}s  llm={model.llm.device}", flush=True)

    gen_cfg = OmniVoiceGenerationConfig(num_step=args.num_step, t_shift=args.t_shift,
                                        guidance_scale=args.guidance_scale)
    prompts = {}
    records = []
    order = {s: i for i, s in enumerate(speakers)}
    torder = {t: i for i, t in enumerate(target_names)}

    for n, (sp, tg) in enumerate(todo, 1):
        if sp not in prompts:
            prompts[sp] = model.create_voice_clone_prompt(
                ref_audio=os.path.join(args.test_dir, f"{sp}.mp3"), ref_text=refs[sp])

        # Seed depends only on the cell, so two checkpoints compared at the same
        # --seed differ by weights alone, not by sampler noise.
        seed = args.seed + 1000 * order[sp] + torder.get(tg, 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if args.device == "mps":
            torch.mps.manual_seed(seed)

        t = time.time()
        audio = model.generate(text=targets[tg], language="en",
                               voice_clone_prompt=prompts[sp],
                               generation_config=gen_cfg)[0]
        dt = time.time() - t

        path = path_for(sp, tg)
        sf.write(path, audio, model.sampling_rate)
        dur = len(audio) / model.sampling_rate
        records.append({"path": os.path.basename(path), "speaker": sp, "target": tg,
                        "text": targets[tg], "seed": seed,
                        "audio_s": round(dur, 3), "gen_s": round(dt, 3),
                        "rtf": round(dt / dur, 4)})
        print(f"  [{n:3d}/{len(todo)}] {sp:9s} x {tg:9s} {dur:5.2f}s  "
              f"RTF {dt/dur:.3f}", flush=True)

    manifest = os.path.join(out, "manifest.json")
    prev = []
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            prev = json.load(f).get("samples", [])
    merged = {r["path"]: r for r in prev}
    merged.update({r["path"]: r for r in records})
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "num_step": args.num_step,
                   "t_shift": args.t_shift, "guidance_scale": args.guidance_scale,
                   "seed": args.seed, "device": args.device, "dtype": args.dtype,
                   "samples": sorted(merged.values(), key=lambda r: r["path"])}, f,
                  indent=2)

    rtfs = [r["rtf"] for r in records]
    print(f"\n{len(records)} clips in {out}/   median RTF {np.median(rtfs):.3f}"
          f"   ({1/np.median(rtfs):.2f}x realtime)")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
