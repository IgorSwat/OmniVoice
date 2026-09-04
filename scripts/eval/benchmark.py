#!/usr/bin/env python3
"""Latency + quality baseline at a controlled operating point.

RTF is meaningless without fixing the sequence geometry --- a fixed prefix cost
amortises over the target, so the same model measures RTF 0.79 on a 2.7 s output
and 0.36 on a 13.2 s one. This pins both sides: reference ~5 s, target ~5 s, so
every later optimisation (prefix blocking, pruning, step reduction) can be
compared against one number rather than against whatever length it happened to
run on.

Pairs are drawn from the held-out dev set. Prompts are built **directly from the
pre-tokenised codec `.npy`**, which is both faster and more faithful to
deployment --- the encoder runs once per voice, offline, and the shipped device
never has it (plan section 3). Because the duration estimator scales the
reference's speaking rate by a character-weight ratio rather than predicting in
absolute terms, target texts are selected per reference so the *estimated*
duration actually lands in the requested window.

Reported per config: end-to-end latency and RTF, the LM/codec/other split, the
sequence length actually seen by the backbone, and WER against the target text.

    python scripts/benchmark.py                       # 32-step baseline
    python scripts/benchmark.py --config 32:0.1 --config 16:0.1
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DTYPES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
FRAME_RATE = 25.0


def parse_config(spec):
    """``"16:0.1"`` or ``"16:0.1:0.0"`` (the third field is guidance_scale)."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"bad --config {spec!r}: want STEPS:T_SHIFT[:GUIDANCE[:k=v...]]")
    cfg = {"num_step": int(parts[0]), "t_shift": float(parts[1])}
    tag = f"s{cfg['num_step']}_t{cfg['t_shift']}"
    if len(parts) >= 3:
        cfg["guidance_scale"] = float(parts[2])
        tag += f"_g{cfg['guidance_scale']}"
    for extra in parts[3:]:                      # e.g. remask_ratio=0.5
        k, v = extra.split("=", 1)
        cfg[k] = int(v) if v.lstrip("-").isdigit() else float(v)
        tag += f"_{''.join(w[0] for w in k.split('_'))}{v}"
    return tag, cfg


def build_pairs(args, estimator):
    """Cross-speaker (reference, target-text) pairs landing in the duration window."""
    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="|"):
            d, b = os.path.dirname(r["name"]), os.path.basename(r["name"])
            p = os.path.join(args.data_root, d, "codecs", f"{b}.npy")
            if os.path.exists(p):
                r["_codec"] = p
                r["_frames"] = int(np.load(p, mmap_mode="r").shape[-1])
                rows.append(r)

    lo, hi = args.dur_range
    refs = [r for r in rows if lo <= r["_frames"] / FRAME_RATE <= hi]
    rng = random.Random(args.seed)
    rng.shuffle(refs)

    pairs = []
    for ref in refs:
        if len(pairs) >= args.n:
            break
        # A target text is only "~5 s" relative to this reference's speaking rate.
        cands = [
            t for t in rows
            if t["speaker_id"] != ref["speaker_id"]
            and lo <= estimator.estimate_duration(
                t["transcription"], ref["transcription"], ref["_frames"]) / FRAME_RATE <= hi
        ]
        if not cands:
            continue
        tgt = rng.choice(cands)
        pairs.append((ref, tgt))
    return pairs


class Split:
    """Wall-clock split of one generate() call, with MPS synchronised boundaries."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self._sync = (lambda: __import__("torch").mps.synchronize()) if device == "mps" else (lambda: None)
        self.reset()
        self._orig_llm = model.llm.forward
        self._orig_dec = model.audio_tokenizer.decode

        def llm_fwd(*a, **kw):
            self._sync(); t = time.perf_counter()
            out = self._orig_llm(*a, **kw)
            self._sync(); self.llm += time.perf_counter() - t
            self.llm_calls += 1
            h = getattr(out, "last_hidden_state", None)
            if h is not None:
                self.batch, self.seqlen = h.shape[0], h.shape[1]
            return out

        def decode(*a, **kw):
            t = time.perf_counter()
            out = self._orig_dec(*a, **kw)
            self.codec += time.perf_counter() - t
            return out

        model.llm.forward = llm_fwd
        model.audio_tokenizer.decode = decode

    def reset(self):
        self.llm = self.codec = 0.0
        self.llm_calls = 0
        self.batch = self.seqlen = 0

    def restore(self):
        self.model.llm.forward = self._orig_llm
        self.model.audio_tokenizer.decode = self._orig_dec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/dev_set.csv")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="runs/benchmark")
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--config", action="append", metavar="STEPS:T_SHIFT")
    ap.add_argument("--dur-range", type=float, nargs=2, default=[4.5, 5.5],
                    metavar=("LO", "HI"), help="seconds, applied to BOTH sides")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--asr-model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--no-wer", dest="wer", action="store_false")
    ap.add_argument("--prefix-cached", action="store_true",
                    help="prefix K/V hoist (bit-identical output, real speedup)")
    ap.add_argument("--prefix-blocked", action="store_true",
                    help="apply the stage-2 attention block in the stock path. "
                         "REQUIRED for any p2 descendant when not using "
                         "--prefix-cached, which blocks implicitly. Without it "
                         "the model runs under a topology it was not trained "
                         "for and the numbers are invalid.")
    args = ap.parse_args()

    import torch
    import soundfile as sf
    from omnivoice.models.omnivoice import (OmniVoice, OmniVoiceGenerationConfig,
                                            VoiceClonePrompt, place_codec)

    configs = [parse_config(c) for c in (args.config or ["32:0.1"])]
    os.makedirs(args.out_dir, exist_ok=True)

    print("loading model ...", flush=True)
    dtype = getattr(torch, DTYPES[args.dtype])
    model = OmniVoice.from_pretrained(args.model, device_map="cpu", dtype=torch.float32)
    codec = model.audio_tokenizer
    model.audio_tokenizer = None
    model.to(args.device, dtype)
    # The codec must not inherit the LM's dtype (fp16 overflows to NaN) and is
    # probed onto the device rather than assumed to work there.
    model.audio_tokenizer = place_codec(codec, args.device,
                                        model.config.num_audio_codebook)
    model.eval()

    pairs = build_pairs(args, model.duration_estimator)
    if len(pairs) < args.n:
        print(f"warning: only {len(pairs)} pairs available (asked {args.n})")
    print(f"{len(pairs)} pairs, reference and target both in "
          f"[{args.dur_range[0]}, {args.dur_range[1]}]s\n")

    # Prompts from codec tokens; ref_rms recovered by decoding the reference.
    prompts = {}
    for ref, _ in pairs:
        if ref["name"] in prompts:
            continue
        toks = torch.from_numpy(np.load(ref["_codec"])).long()
        with torch.no_grad():
            wav = model.audio_tokenizer.decode(
                toks.unsqueeze(0).to(model.audio_tokenizer.device)).audio_values[0]
        wav = wav.detach().cpu().numpy()
        prompts[ref["name"]] = VoiceClonePrompt(
            ref_audio_tokens=toks, ref_text=ref["transcription"],
            ref_rms=float(np.sqrt(np.mean(wav ** 2))))

    if args.prefix_cached:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import prefix_cache
        # guidance_scale=0 makes the unconditional branch dead weight; skipping
        # it is where a guidance-distilled student's speedup actually is.
        skip_u = all(c.get("guidance_scale", 2.0) == 0 for _, c in configs)
        prefix_cache.enable(model, skip_uncond=skip_u)
        if skip_u:
            print("  unconditional branch SKIPPED (guidance_scale=0)")
        print("prefix K/V cache enabled")

    split = Split(model, args.device)
    sync = split._sync
    results = {}

    for tag, cfg in configs:
        gc = OmniVoiceGenerationConfig(prefix_blocked=args.prefix_blocked, **cfg)
        recs = []
        for i, (ref, tgt) in enumerate(pairs[:args.warmup] + pairs):
            warm = i < args.warmup
            split.reset()
            sync(); t0 = time.perf_counter()
            audio = model.generate(text=tgt["transcription"], language="en",
                                   voice_clone_prompt=prompts[ref["name"]],
                                   generation_config=gc)[0]
            sync(); total = time.perf_counter() - t0
            if warm:
                continue
            dur = len(audio) / model.sampling_rate
            path = os.path.join(args.out_dir, tag, f"{len(recs):03d}.wav")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sf.write(path, audio, model.sampling_rate)
            recs.append({
                "path": path, "text": tgt["transcription"],
                "ref": ref["name"], "ref_s": ref["_frames"] / FRAME_RATE,
                "audio_s": dur, "total_s": total, "llm_s": split.llm,
                "codec_s": split.codec, "other_s": total - split.llm - split.codec,
                "llm_calls": split.llm_calls, "seqlen": split.seqlen,
                "batch": split.batch, "rtf": total / dur,
            })
        results[tag] = recs
        print(f"{tag}: {len(recs)} timed samples", flush=True)

    split.restore()

    if args.wer:
        import jiwer, mlx_whisper
        from whisper_normalizer.english import EnglishTextNormalizer
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "p1_step_reduction"))
        from wer_sweep import ascii_punct
        norm = EnglishTextNormalizer()
        for tag, recs in list(results.items()):
            S = D = I = N = 0
            for r in recs:
                hyp = mlx_whisper.transcribe(r["path"], path_or_hf_repo=args.asr_model,
                                             language="en", verbose=None)["text"]
                m = jiwer.process_words(norm(ascii_punct(r["text"])), norm(ascii_punct(hyp)))
                S += m.substitutions; D += m.deletions; I += m.insertions
                N += m.substitutions + m.deletions + m.hits
            results[tag + "_wer"] = {"wer": (S + D + I) / N, "S": S, "D": D, "I": I, "N": N}

    # ---- report -----------------------------------------------------------
    def q(v, p):
        return float(np.percentile(v, p))

    print(f"\n{'='*78}\nBASELINE  --  reference {args.dur_range[0]}-{args.dur_range[1]}s, "
          f"target {args.dur_range[0]}-{args.dur_range[1]}s\n"
          f"{args.device} / {args.dtype}, CFG on (guidance_scale=2.0)\n{'='*78}")
    for tag, _ in configs:
        r = results[tag]
        tot = np.array([x["total_s"] for x in r])
        rtf = np.array([x["rtf"] for x in r])
        llm = np.array([x["llm_s"] for x in r])
        cod = np.array([x["codec_s"] for x in r])
        oth = np.array([x["other_s"] for x in r])
        aud = np.array([x["audio_s"] for x in r])
        calls = int(np.median([x["llm_calls"] for x in r]))
        print(f"\n--- {tag}  (n={len(r)}) ---")
        print(f"  audio out      {aud.mean():6.2f}s   reference "
              f"{np.mean([x['ref_s'] for x in r]):.2f}s   "
              f"seqlen {int(np.median([x['seqlen'] for x in r]))} tok "
              f"(batch {int(np.median([x['batch'] for x in r]))} = CFG)")
        print(f"  latency        {tot.mean():6.2f}s   "
              f"p50 {q(tot,50):.2f}  p90 {q(tot,90):.2f}")
        print(f"  RTF            {rtf.mean():6.3f}    "
              f"p50 {q(rtf,50):.3f}  p90 {q(rtf,90):.3f}   "
              f"({1/rtf.mean():.2f}x realtime)")
        print(f"  LM forward     {llm.mean():6.2f}s  {llm.mean()/tot.mean()*100:5.1f}%   "
              f"{calls} calls, {llm.mean()/calls*1000:.1f} ms/call")
        print(f"  codec decode   {cod.mean():6.2f}s  {cod.mean()/tot.mean()*100:5.1f}%")
        print(f"  other          {oth.mean():6.2f}s  {oth.mean()/tot.mean()*100:5.1f}%")
        if tag + "_wer" in results:
            w = results[tag + "_wer"]
            print(f"  WER            {w['wer']*100:6.2f}%   "
                  f"S{w['S']} D{w['D']} I{w['I']} / {w['N']} words")

    out = os.path.join(args.out_dir, "report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
