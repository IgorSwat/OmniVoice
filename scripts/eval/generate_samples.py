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

`--prefix-blocked` runs generation with prefix-query -> target-key attention cut,
which is the topology a prefix-blocked checkpoint was trained for. Without it,
such a checkpoint is evaluated in the one configuration it was trained NOT to be
used in.

    # the three arms that decide whether stage 2 works
    python scripts/eval/generate_samples.py --cross                      # teacher, full
    python scripts/eval/generate_samples.py --cross --prefix-blocked     # teacher, blocked (zero-shot)
    python scripts/eval/generate_samples.py --cross --prefix-blocked \
        --model runs/prefix_blocked                                      # student, blocked

NOTE: this changes the attention MASK only. It does not hoist the prefix forward
out of the step loop, so RTF is unchanged -- this measures quality, not speed.
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


def enable_prefix_blocking(model):
    """Cut prefix-query -> target-key attention during generation.

    `_generate_iterative` builds a `[2B, 1, S, S]` mask and reuses it across every
    diffusion step, so the block can be applied by wrapping `forward` rather than
    forking the whole sampler.

    The per-item lengths are recoverable from the mask itself. The conditional row
    `i` is filled as `[:c_len, :c_len] = True`, so query 0 has exactly `c_len` keys;
    the unconditional row `B + i` is built target-only as `[:u_len, :u_len] = True`
    with `u_len == target_len`, so its query 0 has exactly `t_len` keys. The prefix
    is therefore `[0, c_len - t_len)` and the target `[c_len - t_len, c_len)`.

    The unconditional branch carries no prefix at all, so it needs no blocking.

    Applied in place: the sampler hands back the same mask tensor every step and
    the write is idempotent, so this costs one pass, not one per step.
    """
    original = model.forward

    def forward(*args, **kwargs):
        am = kwargs.get("attention_mask")
        if am is not None and am.dim() == 4 and am.shape[0] % 2 == 0:
            B = am.shape[0] // 2
            for i in range(B):
                # Must be idempotent: the sampler reuses this tensor every step and
                # the write below is in place. Reading query row 0 would work only
                # once -- row 0 is a PREFIX query, so after the first block it
                # reports `p` rather than `c_len`, and the next call would zero a
                # shifted region. Target-query rows are never modified, and they
                # hold the maximum, so the max over queries is stable.
                c_len = int(am[i, 0].sum(-1).max())
                t_len = int(am[B + i, 0].sum(-1).max())
                p = c_len - t_len
                if 0 < p < c_len:
                    am[i, :, :p, p:c_len] = False
        return original(*args, **kwargs)

    model.forward = forward
    return original


def make_blind(dirs, out_dir, seed):
    """Copy N result directories into randomised A/B/C... sets for blind listening.

    Letters are shuffled independently per row, so no arm sits under one letter
    across the sheet. With more than two arms the sheet asks for a ranking rather
    than a preference.
    """
    if len(dirs) < 2:
        raise SystemExit("--blind needs at least two directories")
    sets = [{n for n in os.listdir(d) if n.endswith(".wav")} for d in dirs]
    names = sorted(set.intersection(*sets))
    if not names:
        raise SystemExit(f"no .wav files common to all of {dirs}")

    os.makedirs(out_dir, exist_ok=True)
    letters = [chr(ord("A") + i) for i in range(len(dirs))]
    rng = random.Random(seed)
    key = {}
    for n in names:
        stem = n[:-4]
        shuffled = letters[:]
        rng.shuffle(shuffled)
        for letter, src in zip(shuffled, dirs):
            dst = f"{stem}_{letter}.wav"
            shutil.copy(os.path.join(src, n), os.path.join(out_dir, dst))
            key[dst] = src

    sheet = os.path.join(out_dir, "listening_sheet.md")
    with open(sheet, "w", encoding="utf-8") as f:
        f.write(f"# Blind test — {len(dirs)} arms\n\n"
                "Same reference, same text, same seed within each row. The letters are "
                "shuffled independently per row, so a letter means nothing across rows.\n\n")
        if len(dirs) == 2:
            f.write("- **prefer** — A, B, or `=` if you cannot tell\n"
                    "- **conf** — 1 (guessing) / 2 (leaning) / 3 (obvious)\n\n"
                    "| pair | prefer | conf | what differed |\n|---|---|---|---|\n")
        else:
            f.write(f"- **rank** — best to worst, e.g. `{'>'.join(letters)}`. "
                    f"Use `=` for ties, e.g. `{letters[0]}={letters[1]}>{letters[2]}`\n"
                    "- **conf** — 1 (guessing) / 2 (leaning) / 3 (obvious)\n\n"
                    "| row | rank | conf | what differed |\n|---|---|---|---|\n")
        for n in names:
            f.write(f"| {n[:-4]:<12s} |  |  |  |\n")

    # The key must not sit next to the audio, or it is one `ls` from spoiling the test.
    key_path = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".",
                            f".{os.path.basename(out_dir.rstrip('/'))}_key.json")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "arms": dirs, "key": key}, f, indent=2)
    print(f"{len(names)} rows x {len(dirs)} arms -> {out_dir}\nsheet: {sheet}\nkey:   {key_path}")


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
    ap.add_argument("--prefix-blocked", action="store_true",
                    help="cut prefix->target attention (mask only; RTF unchanged). "
                         "Required to evaluate a prefix-blocked checkpoint in the "
                         "topology it was trained for.")
    ap.add_argument("--prefix-cached", action="store_true",
                    help="prefix blocking AND the K/V hoist: the prefix runs once, "
                         "each step forwards only the target. Bit-identical output "
                         "to --prefix-blocked, but actually faster.")
    ap.add_argument("--cross", action="store_true",
                    help="every speaker x every target (81) instead of the diagonal (9)")
    ap.add_argument("--speakers", nargs="*", default=None, help="subset of voices")
    ap.add_argument("--targets", nargs="*", default=None, help="subset of target texts")
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate files that already exist")
    ap.add_argument("--blind", nargs="+", metavar="DIR", default=None,
                    help="skip generation; shuffle N existing result dirs into a "
                         "blind A/B/C... set")
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

    suffix = "_cached" if args.prefix_cached else ("_blocked" if args.prefix_blocked else "")
    tag = args.tag or model_tag(args.model) + suffix
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
          f"guidance_scale={args.guidance_scale}"
          f"{', prefix-cached' if args.prefix_cached else (', prefix-blocked' if args.prefix_blocked else '')})")

    t0 = time.time()
    model = load_model(args.model, args.device, args.dtype)
    print(f"loaded in {time.time()-t0:.1f}s  llm={model.llm.device}", flush=True)
    if args.prefix_cached:
        import prefix_cache
        # With guidance_scale=0 the unconditional branch is still computed and
        # then discarded by the sampler; skipping it is where a guidance-distilled
        # student's speedup actually lives.
        skip_u = args.guidance_scale == 0
        prefix_cache.enable(model, skip_uncond=skip_u)
        print("prefix K/V CACHE enabled (prefix runs once per generation)"
              + ("; unconditional branch SKIPPED (guidance_scale=0)" if skip_u else ""),
              flush=True)
    elif args.prefix_blocked:
        enable_prefix_blocking(model)
        print("prefix blocking ENABLED (mask only -- RTF is unchanged)", flush=True)

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
        json.dump({"model": args.model, "prefix_blocked": args.prefix_blocked,
                   "prefix_cached": args.prefix_cached,
                   "num_step": args.num_step,
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
