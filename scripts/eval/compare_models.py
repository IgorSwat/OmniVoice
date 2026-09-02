#!/usr/bin/env python3
"""Score generated arms against a baseline on metrics that mean something.

Listening tests are the ground truth but they are slow, noisy at n=9, and have
already contradicted themselves once in this project. This scores the same audio
automatically on the three axes the distillation stages can actually damage:

  WER       intelligibility -- catches dropped and mangled words, the failure
            that separated guidance_scale=0 so cleanly
  SIM       speaker similarity between the reference clip and the generated
            audio (WavLM x-vector cosine). THE metric for a clone model, and the
            one nothing in this project has measured yet. Same-speaker pairs
            score ~0.97 and different-speaker pairs ~0.45, so the scale is wide
            enough to see real degradation.
  prosody   f0 spread, energy dynamics, speaking rate, pause fraction. Every
            listening round that disliked an arm described it as "flat"; this is
            that complaint as a number.

**The control arm is what makes any of it interpretable.** Generation is
stochastic, so re-running the SAME model at a different seed produces a nonzero
delta on every metric. That delta is the noise floor: a student differs
meaningfully from the teacher only if it exceeds the teacher's disagreement with
itself. Pass the reseeded baseline with --control and the report prints both.

Arms are directories produced by generate_samples.py -- their manifest.json
supplies the speaker, target text and seed, so nothing has to be re-derived.

    python scripts/eval/compare_models.py \\
        --arm teacher=tmp/cmp/teacher --control teacher_reseed=tmp/cmp/teacher_b \\
        --arm p1p2p3=tmp/cmp/p1p2p3 --baseline teacher
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common import TEST_DIR  # noqa: E402

SR = 16000          # both WavLM and the prosody features work at 16 kHz


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class Scorer:
    """Lazily loads the ASR and speaker models; caches per-file results."""

    def __init__(self, asr_model, cache_path=None):
        self.asr_model = asr_model
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)
        self._asr = self._xv = self._fe = None
        self._norm = self._ascii = None

    def save(self):
        if self.cache_path:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)

    def _lazy(self):
        if self._xv is None:
            import torch
            from transformers import AutoFeatureExtractor, WavLMForXVector
            self._fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
            self._xv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
            self._torch = torch
        if self._norm is None:
            from whisper_normalizer.english import EnglishTextNormalizer
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "p1_step_reduction"))
            from wer_sweep import ascii_punct
            self._norm, self._ascii = EnglishTextNormalizer(), ascii_punct

    def embed(self, path):
        key = f"emb::{path}::{os.path.getmtime(path):.0f}"
        if key in self.cache:
            return np.array(self.cache[key])
        self._lazy()
        import librosa
        y, _ = librosa.load(path, sr=SR, mono=True)
        with self._torch.no_grad():
            i = self._fe(y, sampling_rate=SR, return_tensors="pt")
            e = self._torch.nn.functional.normalize(self._xv(**i).embeddings, dim=-1)
        v = e[0].numpy()
        self.cache[key] = v.tolist()
        return v

    def transcribe(self, path):
        key = f"asr::{path}::{os.path.getmtime(path):.0f}"
        if key not in self.cache:
            import mlx_whisper
            self.cache[key] = mlx_whisper.transcribe(
                path, path_or_hf_repo=self.asr_model, language="en",
                verbose=None)["text"].strip()
        return self.cache[key]

    def wer(self, path, ref_text):
        import jiwer
        self._lazy()
        hyp = self.transcribe(path)
        m = jiwer.process_words(self._norm(self._ascii(ref_text)),
                                self._norm(self._ascii(hyp)))
        return (m.substitutions, m.deletions, m.insertions,
                m.substitutions + m.deletions + m.hits, hyp)

    def prosody(self, path):
        """f0 spread in semitones, energy dynamics in dB, rate and pause share.

        f0 is measured in semitones so it is comparable across speakers with
        different registers; a flat delivery compresses this regardless of pitch.
        """
        key = f"pros::{path}::{os.path.getmtime(path):.0f}"
        if key in self.cache:
            return self.cache[key]
        import librosa
        y, _ = librosa.load(path, sr=SR, mono=True)
        f0 = librosa.yin(y, fmin=65, fmax=400, sr=SR, frame_length=1024, hop_length=256)
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        voiced = rms > (0.15 * np.median(rms[rms > 0]) if (rms > 0).any() else 0)
        f0v = f0[voiced & np.isfinite(f0) & (f0 > 0)]
        semis = 12 * np.log2(f0v / np.median(f0v)) if len(f0v) > 8 else np.zeros(1)
        db = 20 * np.log10(np.clip(rms, 1e-6, None))
        out = {
            "f0_std_semitones": float(np.std(semis)),
            "energy_std_db": float(np.std(db[voiced])) if voiced.any() else 0.0,
            "speech_frac": float(voiced.mean()),
            "duration_s": float(len(y) / SR),
        }
        self.cache[key] = out
        return out


# ---------------------------------------------------------------------------


def load_arm(d):
    mf = os.path.join(d, "manifest.json")
    if not os.path.exists(mf):
        raise SystemExit(f"{d} has no manifest.json (generate it with generate_samples.py)")
    with open(mf, encoding="utf-8") as f:
        m = json.load(f)
    return {(r["speaker"], r["target"]): {**r, "abs": os.path.join(d, r["path"])}
            for r in m["samples"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--control", default=None, metavar="NAME=DIR",
                    help="the baseline model re-run at a different seed; its "
                         "deltas are the noise floor every other arm is read against")
    ap.add_argument("--baseline", default=None,
                    help="arm to compare against (default: the first --arm)")
    ap.add_argument("--test-dir", default=TEST_DIR)
    ap.add_argument("--asr-model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--cache", default="runs/compare_cache.json")
    ap.add_argument("--out", default="runs/compare_report.json")
    args = ap.parse_args()

    specs = [a.split("=", 1) for a in args.arm]
    if args.control:
        specs.append(args.control.split("=", 1))
        control_name = args.control.split("=", 1)[0]
    else:
        control_name = None
    arms = {n: load_arm(d) for n, d in specs}
    base = args.baseline or specs[0][0]
    if base not in arms:
        raise SystemExit(f"--baseline {base} is not one of {list(arms)}")

    cells = sorted(set.intersection(*(set(a) for a in arms.values())))
    if not cells:
        raise SystemExit("arms share no (speaker, target) cells")
    print(f"{len(cells)} cells x {len(arms)} arms   baseline={base}"
          f"{'  control=' + control_name if control_name else ''}\n")

    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    sc = Scorer(args.asr_model, args.cache)
    res = {n: {} for n in arms}
    for n, arm in arms.items():
        for k, (sp, tg) in enumerate(cells, 1):
            r = arm[(sp, tg)]
            S, D, I, N, hyp = sc.wer(r["abs"], r["text"])
            ref = os.path.join(args.test_dir, f"{sp}.mp3")
            sim = float(np.dot(sc.embed(ref), sc.embed(r["abs"])))
            res[n][(sp, tg)] = {"S": S, "D": D, "I": I, "N": N, "sim": sim,
                                "hyp": hyp, **sc.prosody(r["abs"])}
            print(f"  {n:16s} {k}/{len(cells)}", end="\r", flush=True)
        sc.save()
    print(" " * 40, end="\r")

    def agg(n):
        v = res[n]
        N = sum(c["N"] for c in v.values()) or 1
        return {
            "wer": sum(c["S"] + c["D"] + c["I"] for c in v.values()) / N,
            "del": sum(c["D"] for c in v.values()) / N,
            "sim": float(np.mean([c["sim"] for c in v.values()])),
            "f0_std_semitones": float(np.mean([c["f0_std_semitones"] for c in v.values()])),
            "energy_std_db": float(np.mean([c["energy_std_db"] for c in v.values()])),
            "duration_s": float(np.mean([c["duration_s"] for c in v.values()])),
        }

    A = {n: agg(n) for n in arms}
    print(f"{'arm':22s} {'WER':>7s} {'del':>7s} {'SIM':>7s} {'f0 st':>7s} "
          f"{'dB std':>7s} {'dur s':>7s}")
    print("-" * 70)
    for n in arms:
        a = A[n]
        print(f"{n:22s} {a['wer']*100:6.2f}% {a['del']*100:6.2f}% {a['sim']:7.4f} "
              f"{a['f0_std_semitones']:7.3f} {a['energy_std_db']:7.2f} {a['duration_s']:7.2f}")

    print(f"\npaired deltas vs {base}   (per-cell, same reference/text/seed)")
    print(f"{'arm':22s} {'dWER':>8s} {'dSIM':>9s} {'d f0 st':>9s} {'d dB std':>9s}")
    print("-" * 62)
    floor = None
    for n in arms:
        if n == base:
            continue
        d = {k: float(np.mean([res[n][c][k] - res[base][c][k] for c in cells]))
             for k in ("sim", "f0_std_semitones", "energy_std_db")}
        dw = A[n]["wer"] - A[base]["wer"]
        tag = "   <- NOISE FLOOR" if n == control_name else ""
        print(f"{n:22s} {dw*100:+7.2f}pp {d['sim']:+9.4f} "
              f"{d['f0_std_semitones']:+9.3f} {d['energy_std_db']:+9.2f}{tag}")
        if n == control_name:
            floor = {**d, "wer": dw}

    if floor:
        print(f"\nsignificance vs the noise floor (|delta| / |floor delta|; >1 means "
              f"larger than the model's\ndisagreement with itself at a different seed):")
        print(f"{'arm':22s} {'WER':>8s} {'SIM':>8s} {'f0':>8s} {'dB':>8s}")
        for n in arms:
            if n in (base, control_name):
                continue
            d = {k: float(np.mean([res[n][c][k] - res[base][c][k] for c in cells]))
                 for k in ("sim", "f0_std_semitones", "energy_std_db")}
            dw = A[n]["wer"] - A[base]["wer"]
            def rel(x, f):
                return abs(x) / abs(f) if abs(f) > 1e-9 else float("inf")
            print(f"{n:22s} {rel(dw, floor['wer']):8.1f} {rel(d['sim'], floor['sim']):8.1f} "
                  f"{rel(d['f0_std_semitones'], floor['f0_std_semitones']):8.1f} "
                  f"{rel(d['energy_std_db'], floor['energy_std_db']):8.1f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"baseline": base, "control": control_name, "aggregate": A,
                   "cells": {n: {f"{s}|{t}": v for (s, t), v in res[n].items()}
                             for n in arms}}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
