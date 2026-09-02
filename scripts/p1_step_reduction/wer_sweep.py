#!/usr/bin/env python3
"""Objective WER sweep over diffusion-sampler schedules.

Blind listening (9 pairs) showed the 32-step schedule winning 3 pairs to 0 with
6 ties, and every decided loss in the 16-step arm was a *discrete* defect --- a
dropped word, a swallowed onset --- rather than general dulling. That is the
signature of a conditional-independence violation at commit time, and it is
exactly what WER measures and what UTMOS/speaker-SIM cannot see.

This script turns that impression into a rate. Every speaker in ``data/test``
says every target in ``targets.txt`` (9 x 9 = 81 cells), under each sampler
config, and the output is transcribed with Whisper and scored against the
target text.

Two design points that make the comparison paired and cheap:

* **Same seed per cell across configs.** A cell's seed depends only on
  (speaker, target), so the config contrast is not confounded by sampler noise.
* **Three resumable stages.** ``generate`` is the expensive one (~5 s/utterance);
  ``asr`` and ``report`` re-run for free off the on-disk artifacts, so adding a
  config or changing the metric never re-synthesises anything.

Because the observed failure mode is *dropped words*, the report breaks WER into
substitutions / deletions / insertions separately --- an elevated deletion rate
is the specific prediction under test, and a raw WER number would hide it.

    # the two schedules from the listening test
    python scripts/step_reduction/wer_sweep.py --config 32:0.1 --config 16:0.2

    # t_shift is free to sweep; add rungs without regenerating the old ones
    python scripts/step_reduction/wer_sweep.py \
        --config 32:0.1 --config 16:0.2 --config 16:0.3 --config 16:0.5
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DTYPES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}

# The reference recording for this slot is a real, identifiable public figure and
# the target text as written is campaign-style rhetoric. Under the cross-product
# design every voice speaks every target, so that pairing would be synthesised
# directly. WER is indifferent to *what* is said --- only to whether the words
# survive --- so a neutral line of similar length and prosodic demand costs the
# experiment nothing. Delete this dict to use targets.txt verbatim.
TARGET_OVERRIDES = {
    "trump": (
        "The weather has been changing so quickly this week. One morning it is "
        "warm and sunny, and by the afternoon you need a jacket again."
    ),
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def read_inputs(test_dir):
    """Return (ref_texts, target_texts) keyed by speaker / target name."""
    refs = {}
    with open(os.path.join(test_dir, "transcripts.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, text = line.split("|", 1)
            refs[name.strip()] = text.strip()

    targets = {}
    with open(os.path.join(test_dir, "targets.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, text = line.split(":", 1)
            targets[name.strip().lower()] = text.strip()

    targets.update({k: v for k, v in TARGET_OVERRIDES.items() if k in targets})

    missing = [s for s in refs if not os.path.exists(os.path.join(test_dir, f"{s}.mp3"))]
    if missing:
        raise SystemExit(f"missing reference audio for: {missing}")
    return refs, targets


def parse_config(spec):
    """``"16:0.2"`` -> ``("s16_t0.2", {"num_step": 16, "t_shift": 0.2})``."""
    steps, t_shift = spec.split(":")
    steps, t_shift = int(steps), float(t_shift)
    return f"s{steps}_t{t_shift}", {"num_step": steps, "t_shift": t_shift}


# Whisper's EnglishTextNormalizer expands contractions only on ASCII apostrophes.
# targets.txt is typed with typographic quotes, so "There’s" normalises to
# "there s" while the ASR's "There's" becomes "there is" --- scoring a perfect
# transcription as an error, once per contraction. Fold the Unicode punctuation
# down to ASCII before normalising.
_PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'", "\u00b4": "'", "`": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2013": " ", "\u2014": " ", "\u2012": " ", "\u2015": " ",
    "\u2026": " ", "\u00a0": " ",
})


def ascii_punct(text):
    return text.translate(_PUNCT_MAP)


def cell_seed(base, speaker_idx, target_idx, rep=0):
    """Seed depends only on the cell, so configs are paired sample-for-sample."""
    return base + 100000 * rep + 1000 * speaker_idx + target_idx


def cell_path(out_dir, tag, speaker, target, rep):
    """Repeat 0 keeps the un-suffixed name, so raising --seeds is purely additive."""
    stem = f"{speaker}__{target}" if rep == 0 else f"{speaker}__{target}__r{rep}"
    return os.path.join(out_dir, "audio", tag, f"{stem}.wav")


# ---------------------------------------------------------------------------
# Stage 1 — generation
# ---------------------------------------------------------------------------


def stage_generate(args, refs, targets, configs):
    import torch
    import soundfile as sf

    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig

    speakers = sorted(refs)
    target_names = sorted(targets)

    todo = []
    for tag, _ in configs:
        for si, sp in enumerate(speakers):
            for ti, tg in enumerate(target_names):
                for rep in range(args.seeds):
                    path = cell_path(args.out_dir, tag, sp, tg, rep)
                    if not (args.resume and os.path.exists(path)):
                        todo.append((tag, si, sp, ti, tg, rep, path))

    if not todo:
        print("generate: nothing to do (all outputs present)")
        return

    print(f"generate: {len(todo)} utterances "
          f"({len(configs)} configs x {len(speakers)} speakers x "
          f"{len(target_names)} targets x {args.seeds} seeds)")

    print("loading model ...", flush=True)
    t0 = time.time()
    dtype = getattr(torch, DTYPES[args.dtype])
    model = OmniVoice.from_pretrained(args.model, device_map="cpu", dtype=torch.float32)
    if args.device == "mps":
        # The Higgs codec cannot run on MPS (output channels > 65536), and
        # loading straight onto MPS segfaults; move the LM only, keep codec on CPU.
        codec, fe = model.audio_tokenizer, model.feature_extractor
        model.audio_tokenizer = None
        model.to(args.device, dtype)
        model.audio_tokenizer, model.feature_extractor = codec, fe
    else:
        model.to(args.device, dtype)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s  llm={model.llm.device}", flush=True)

    prompts = {}
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    cfg_by_tag = dict(configs)

    with open(manifest_path, "a", encoding="utf-8") as mf:
        for n, (tag, si, sp, ti, tg, rep, path) in enumerate(todo, 1):
            if sp not in prompts:
                prompts[sp] = model.create_voice_clone_prompt(
                    ref_audio=os.path.join(args.test_dir, f"{sp}.mp3"), ref_text=refs[sp]
                )

            seed = cell_seed(args.seed, si, ti, rep)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if args.device == "mps":
                torch.mps.manual_seed(seed)

            t = time.time()
            audio = model.generate(
                text=targets[tg],
                language="en",
                voice_clone_prompt=prompts[sp],
                generation_config=OmniVoiceGenerationConfig(**cfg_by_tag[tag]),
            )[0]
            dt = time.time() - t

            os.makedirs(os.path.dirname(path), exist_ok=True)
            sf.write(path, audio, model.sampling_rate)
            dur = len(audio) / model.sampling_rate

            mf.write(json.dumps({
                "path": os.path.relpath(path, args.out_dir),
                "config": tag, "speaker": sp, "target": tg, "rep": rep,
                "text": targets[tg], "seed": seed,
                "audio_s": round(dur, 3), "gen_s": round(dt, 3),
                "rtf": round(dt / dur, 4),
            }) + "\n")
            mf.flush()

            if n % 10 == 0 or n == len(todo):
                print(f"  [{n:4d}/{len(todo)}] {tag:10s} {sp:8s} x {tg:8s} "
                      f"{dur:5.2f}s  RTF {dt/dur:.3f}", flush=True)


# ---------------------------------------------------------------------------
# Stage 2 — ASR
# ---------------------------------------------------------------------------


def stage_asr(args):
    import mlx_whisper

    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    asr_path = os.path.join(args.out_dir, "asr.jsonl")

    rows = dedup_manifest(manifest_path)
    done = set()
    if args.resume and os.path.exists(asr_path):
        with open(asr_path, encoding="utf-8") as f:
            done = {json.loads(line)["path"] for line in f if line.strip()}

    todo = [r for r in rows if r["path"] not in done]
    if not todo:
        print("asr: nothing to do")
        return
    print(f"asr: transcribing {len(todo)} files with {args.asr_model}")

    with open(asr_path, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            wav = os.path.join(args.out_dir, r["path"])
            t = time.time()
            out = mlx_whisper.transcribe(
                wav, path_or_hf_repo=args.asr_model, language="en", verbose=None
            )
            f.write(json.dumps({
                "path": r["path"], "hyp": out["text"].strip(),
                "asr_s": round(time.time() - t, 3),
            }) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"  [{n:4d}/{len(todo)}] {r['path']}", flush=True)


def dedup_manifest(path):
    """Last write wins, so a regenerated cell supersedes its earlier entry."""
    if not os.path.exists(path):
        raise SystemExit(f"no manifest at {path} -- run --stage generate first")
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["path"]] = r
    return list(rows.values())


# ---------------------------------------------------------------------------
# Stage 3 — scoring
# ---------------------------------------------------------------------------


def stage_report(args):
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = EnglishTextNormalizer()
    rows = dedup_manifest(os.path.join(args.out_dir, "manifest.jsonl"))
    asr_path = os.path.join(args.out_dir, "asr.jsonl")
    if not os.path.exists(asr_path):
        raise SystemExit("no asr.jsonl -- run --stage asr first")

    hyps = {}
    with open(asr_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                hyps[r["path"]] = r["hyp"]

    cells = []
    for r in rows:
        if r["path"] not in hyps:
            continue
        ref = norm(ascii_punct(r["text"]))
        hyp = norm(ascii_punct(hyps[r["path"]]))
        if not ref:
            continue
        m = jiwer.process_words(ref, hyp)
        n_ref = m.substitutions + m.deletions + m.hits
        cells.append({
            "config": r["config"], "speaker": r["speaker"], "target": r["target"],
            "rep": r.get("rep", 0),
            "S": m.substitutions, "D": m.deletions, "I": m.insertions, "N": n_ref,
            "wer": (m.substitutions + m.deletions + m.insertions) / n_ref,
            "rtf": r.get("rtf"), "hyp": hyps[r["path"]], "ref": r["text"],
        })

    configs = sorted({c["config"] for c in cells}, key=lambda t: (-int(t[1:].split("_")[0]), t))
    report = {"n_cells": len(cells), "configs": {}}

    def agg(subset):
        N = sum(c["N"] for c in subset) or 1
        S, D, I = (sum(c[k] for c in subset) for k in "SDI")
        return {"wer": (S + D + I) / N, "sub": S / N, "del": D / N, "ins": I / N,
                "N": N, "n": len(subset)}

    print(f"\n{'='*74}\nCorpus WER  ({len(cells)} cells, "
          f"Whisper EnglishTextNormalizer)\n{'='*74}")
    print(f"{'config':12s} {'n':>4s} {'WER':>8s} {'sub':>8s} {'del':>8s} {'ins':>8s} {'RTF':>7s}")
    for tag in configs:
        sub = [c for c in cells if c["config"] == tag]
        a = agg(sub)
        rtfs = [c["rtf"] for c in sub if c["rtf"]]
        report["configs"][tag] = {**a, "rtf_median": float(np.median(rtfs)) if rtfs else None}
        print(f"{tag:12s} {a['n']:4d} {a['wer']*100:7.2f}% {a['sub']*100:7.2f}% "
              f"{a['del']*100:7.2f}% {a['ins']*100:7.2f}% "
              f"{np.median(rtfs) if rtfs else float('nan'):7.3f}")

    # Paired contrast against the first config (the baseline schedule).
    if len(configs) > 1:
        base = configs[0]
        by_cell = defaultdict(dict)
        for c in cells:
            by_cell[(c["speaker"], c["target"], c["rep"])][c["config"]] = c

        print(f"\n{'='*74}\nPaired vs {base}   (per-cell, same seed both arms)\n{'='*74}")
        print(f"{'config':12s} {'pairs':>6s} {'dWER':>8s} {'better':>7s} "
              f"{'worse':>6s} {'tie':>5s} {'sign p':>8s} {'dWER 95% CI':>20s}")
        report["paired"] = {}
        for tag in configs[1:]:
            pairs = [(v[base], v[tag]) for v in by_cell.values() if base in v and tag in v]
            if not pairs:
                continue
            better = sum(1 for a, b in pairs if b["wer"] < a["wer"])
            worse = sum(1 for a, b in pairs if b["wer"] > a["wer"])
            tie = len(pairs) - better - worse
            d = agg([b for _, b in pairs])["wer"] - agg([a for a, _ in pairs])["wer"]
            lo, hi = bootstrap_delta(pairs, args.bootstrap, args.seed)
            p = sign_test(better, worse)
            report["paired"][tag] = {"vs": base, "delta_wer": d, "ci95": [lo, hi],
                                     "better": better, "worse": worse, "tie": tie,
                                     "sign_p": p, "pairs": len(pairs)}
            print(f"{tag:12s} {len(pairs):6d} {d*100:+7.2f}pp {better:7d} {worse:6d} "
                  f"{tie:5d} {p:8.4f}   [{lo*100:+6.2f}, {hi*100:+6.2f}]pp")

    # Where the errors concentrate.
    for axis in ("speaker", "target"):
        print(f"\n{'-'*74}\nWER by {axis}\n{'-'*74}")
        keys = sorted({c[axis] for c in cells})
        print(f"{axis:12s} " + " ".join(f"{t:>12s}" for t in configs))
        rows_out = {}
        for k in keys:
            vals = [agg([c for c in cells if c[axis] == k and c["config"] == t])["wer"]
                    for t in configs]
            rows_out[k] = dict(zip(configs, vals))
            print(f"{k:12s} " + " ".join(f"{v*100:11.2f}%" for v in vals))
        report[f"by_{axis}"] = rows_out

    worst = sorted(cells, key=lambda c: -c["wer"])[:args.show_worst]
    if worst:
        print(f"\n{'-'*74}\nWorst {len(worst)} cells\n{'-'*74}")
        for c in worst:
            print(f"[{c['wer']*100:6.1f}%  S{c['S']} D{c['D']} I{c['I']}]  "
                  f"{c['config']}  {c['speaker']} x {c['target']}")
            print(f"    ref: {c['ref']}")
            print(f"    hyp: {c['hyp']}")

    out = os.path.join(args.out_dir, "report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


def bootstrap_delta(pairs, n_boot, seed):
    """Percentile CI on the corpus-level WER delta, resampling cells."""
    if n_boot <= 0 or not pairs:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a_err = np.array([p[0]["S"] + p[0]["D"] + p[0]["I"] for p in pairs], float)
    a_n = np.array([p[0]["N"] for p in pairs], float)
    b_err = np.array([p[1]["S"] + p[1]["D"] + p[1]["I"] for p in pairs], float)
    b_n = np.array([p[1]["N"] for p in pairs], float)
    idx = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
    d = b_err[idx].sum(1) / b_n[idx].sum(1) - a_err[idx].sum(1) / a_n[idx].sum(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def sign_test(better, worse):
    """Two-sided exact binomial p over decided pairs; ties are uninformative."""
    n = better + worse
    if n == 0:
        return float("nan")
    from math import comb
    k = min(better, worse)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--out-dir", default="runs/step_reduction")
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--config", action="append", metavar="STEPS:T_SHIFT",
                    help="repeatable; first one is the paired baseline "
                         "(default: 32:0.1 and 16:0.2)")
    ap.add_argument("--stage", default="all",
                    choices=["all", "generate", "asr", "report"])
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--asr-model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeats per cell; repeat 0 reuses existing files")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--show-worst", type=int, default=10)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="regenerate/retranscribe even if outputs exist")
    args = ap.parse_args()

    configs = [parse_config(c) for c in (args.config or ["32:0.1", "16:0.2"])]
    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage in ("all", "generate"):
        refs, targets = read_inputs(args.test_dir)
        stage_generate(args, refs, targets, configs)
    if args.stage in ("all", "asr"):
        stage_asr(args)
    if args.stage in ("all", "report"):
        stage_report(args)


if __name__ == "__main__":
    main()
