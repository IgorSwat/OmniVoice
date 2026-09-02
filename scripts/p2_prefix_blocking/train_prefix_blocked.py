#!/usr/bin/env python3
"""Teach a student to run with prefix-blocked attention, matching a full-attention teacher.

The voice-clone sequence is ``[style | text | ref_audio | target]``. Everything
before the target is static across the whole decode, but because attention is
fully bidirectional the prefix's hidden states depend on the target, so all of it
is recomputed at every diffusion step.

Blocking prefix-query -> target-key attention makes the prefix self-contained: it
can be computed once, its K/V cached, and each step then processes only the
target positions. Measured on this checkpoint that is ~2.1x end-to-end at 16
steps for a 6 s reference.

Applying the blocked mask to the teacher directly costs +0.013 weighted CE
(99.6% of its information retained), so the student starts close. This script
closes the rest: teacher runs with FULL attention, student with BLOCKED, and the
student is trained to match the teacher's output distribution.

    python scripts/train_prefix_blocked.py --out-dir runs/prefix_blocked \
        --device cuda --steps 2000 --batch-tokens 8192

Start from an already-width-pruned model with ``--init-from runs/.../rung_704``;
the default starts from the teacher, which isolates the topology change.
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer  # noqa: E402

from omnivoice.models.omnivoice import (  # noqa: E402
    OmniVoice,
    _combine_text,
    _resolve_model_path,
    _tokenize_with_nonverbal_tags,
)
from p4_width_pruning.manifest import codec_path  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# ---------------------------------------------------------------------------
# Data: same-speaker clone pairs in the inference layout
# ---------------------------------------------------------------------------


def read_manifest(path):
    """Pipe-separated, NOT quoted CSV — a literal `"` must not start a quoted field."""
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE))


class ClonePairDataset(Dataset):
    """Pairs of utterances by the same speaker: one is the reference, one the target.

    This mirrors inference rather than the training processor. The processor
    builds its clone prompt as a *prefix of the same utterance*; real voice
    cloning supplies a separate reference clip with its own transcript, which is
    the layout the prefix block is meant to exploit.
    """

    def __init__(self, manifest, data_root="data", ref_manifest=None,
                 min_frames=50, max_frames=750, ref_cap=150, limit=0,
                 only_speakers=None, exclude_speakers=None):
        rows = manifest if isinstance(manifest, list) else read_manifest(manifest)
        if limit:
            rows = rows[:limit]
        # References may come from a different split -- most dev speakers have a
        # single utterance, so pairing the dev set against itself yields almost
        # no evaluable targets. The reference pool therefore holds the *other*
        # utterances by the same speakers, which the speaker holdout has removed
        # from training (see --no-speaker-holdout).
        if ref_manifest is None:
            ref_rows = rows
        else:
            ref_rows = (ref_manifest if isinstance(ref_manifest, list)
                        else read_manifest(ref_manifest))
        if only_speakers is not None:
            rows = [r for r in rows if r["speaker_id"] in only_speakers]
            ref_rows = [r for r in ref_rows if r["speaker_id"] in only_speakers]
        if exclude_speakers:
            rows = [r for r in rows if r["speaker_id"] not in exclude_speakers]
            ref_rows = [r for r in ref_rows if r["speaker_id"] not in exclude_speakers]
        by_spk = defaultdict(list)
        for r in ref_rows:
            by_spk[r["speaker_id"]].append(r)
        self.rows, self.by_spk = rows, by_spk
        self.targets = [
            r for r in rows
            if any(x["name"] != r["name"] for x in by_spk[r["speaker_id"]])
        ]
        self.data_root, self.ref_cap = data_root, ref_cap
        self.min_frames, self.max_frames = min_frames, max_frames

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        tgt = self.targets[i]
        pool = [r for r in self.by_spk[tgt["speaker_id"]] if r["name"] != tgt["name"]]
        ref = pool[random.randrange(len(pool))]
        t = np.load(codec_path(tgt["name"], self.data_root)).astype(np.int64)
        r = np.load(codec_path(ref["name"], self.data_root)).astype(np.int64)
        return {
            "target_codec": torch.from_numpy(t[:, : self.max_frames]),
            "ref_codec": torch.from_numpy(r[:, : self.ref_cap]),
            "text": tgt["transcription"],
            "ref_text": ref["transcription"],
            "lang": tgt.get("language", "en"),
        }


class CloneCollator:
    """Builds `[style | text | ref_audio | target]` exactly as inference does.

    Replicates ``OmniVoice._prepare_inference_inputs`` (which cannot be reused
    here because it allocates on the model's device), then fills the target with
    ground-truth codec and masks a fraction of it — the training objective, in
    the inference layout.
    """

    def __init__(self, tokenizer, num_codebooks, mask_id, denoise=True):
        self.tok = tokenizer
        self.C = num_codebooks
        self.mask_id = mask_id
        self.denoise = denoise

    def _prefix_ids(self, s):
        """Returns (prefix ids [C, P], index where the audio region begins)."""
        style = "<|denoise|>" if self.denoise else ""
        style += f"<|lang_start|>{s['lang']}<|lang_end|>"
        style += "<|instruct_start|>None<|instruct_end|>"
        style_ids = self.tok(style, return_tensors="pt").input_ids.repeat(self.C, 1)
        full = _combine_text(ref_text=s["ref_text"], text=s["text"])
        text_ids = _tokenize_with_nonverbal_tags(
            f"<|text_start|>{full}<|text_end|>", self.tok
        ).repeat(self.C, 1)
        audio_start = style_ids.shape[1] + text_ids.shape[1]
        return torch.cat([style_ids, text_ids, s["ref_codec"]], dim=1), audio_start

    def __call__(self, samples):
        built = []
        for s in samples:
            pre, audio_start = self._prefix_ids(s)
            tgt = s["target_codec"]
            T = tgt.shape[1]
            ratio = random.uniform(0.0, 1.0)          # diffusion noise level
            tm = torch.rand(self.C, T) < ratio
            if tm.sum() == 0:                          # never yield an empty loss
                tm[random.randrange(self.C), random.randrange(T)] = True
            inp = tgt.clone()
            inp[tm] = self.mask_id
            lab = torch.full((self.C, T), -100, dtype=torch.long)
            lab[tm] = tgt[tm]
            built.append((torch.cat([pre, inp], dim=1), lab, pre.shape[1], audio_start))

        B = len(built)
        S = max(x[0].shape[1] for x in built)
        ids = torch.full((B, self.C, S), self.mask_id, dtype=torch.long)
        labels = torch.full((B, self.C, S), -100, dtype=torch.long)
        audio_mask = torch.zeros(B, S, dtype=torch.bool)
        valid = torch.zeros(B, S, dtype=torch.bool)
        prefix_len = torch.zeros(B, dtype=torch.long)
        for b, (x, lab, P, audio_start) in enumerate(built):
            L = x.shape[1]
            ids[b, :, :L] = x
            labels[b, :, P:L] = lab
            # audio region spans ref_audio + target, i.e. everything after the text
            audio_mask[b, audio_start:L] = True
            valid[b, :L] = True
            prefix_len[b] = P
        return {
            "input_ids": ids,
            "labels": labels,
            "audio_mask": audio_mask,
            "valid": valid,
            "prefix_len": prefix_len,
            "position_ids": torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
        }


def build_masks(valid, prefix_len, block):
    """4D attention masks. `block` cuts prefix-query -> target-key edges only."""
    B, S = valid.shape
    m = valid[:, None, None, :].expand(B, 1, S, S).clone()
    if block:
        pos = torch.arange(S, device=valid.device)
        is_prefix = pos[None, :] < prefix_len[:, None]           # [B, S]
        is_target = valid & ~is_prefix
        m &= ~(is_prefix[:, None, :, None] & is_target[:, None, None, :])
    return m.contiguous()


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------


def logits_of(model, batch, attn):
    e = model._prepare_embed_inputs(batch["input_ids"], batch["audio_mask"])
    h = model.llm(
        inputs_embeds=e,
        attention_mask=attn,
        position_ids=batch["position_ids"],
        return_dict=True,
    ).last_hidden_state
    b, s, _ = h.shape
    return model.audio_heads(h).view(
        b, s, model.config.num_audio_codebook, model.config.audio_vocab_size
    ).permute(0, 2, 1, 3)


def weighted_kd(s_logits, t_logits, labels, w, temperature=1.0):
    """Codebook-weighted KL(teacher || student) on loss-bearing positions."""
    mask = (labels != -100).float()
    t = temperature
    lps = torch.log_softmax(s_logits / t, dim=-1)
    lpt = torch.log_softmax(t_logits / t, dim=-1)
    kl = (lpt.exp() * (lpt - lps)).sum(dim=-1)
    per_cb = (kl * mask).sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp(min=1.0)
    return (per_cb * w).sum() * t * t


def weighted_ce(logits, labels, w):
    pt = torch.nn.functional.cross_entropy(
        logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=-100
    )
    v = (labels != -100).float()
    per_cb = (pt * v).sum(dim=(0, 2)) / v.sum(dim=(0, 2)).clamp(min=1.0)
    return (per_cb * w).sum()


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(student, teacher, loader, device, w, seed=1234, max_batches=0):
    """Held-out CE for both models plus the student's KL to the teacher.

    KL is the metric that actually moves: measured on this checkpoint, CE
    separates teacher-full / teacher-blocked / student-blocked by under 0.4%,
    while KL separates them by 2-3x. Reading CE alone makes a working run and a
    dead one look identical.

    RNG is pinned and restored so the random mask ratios are the same at every
    eval, making successive numbers comparable.
    """
    py, th = random.getstate(), torch.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)
    student.eval()
    tot_s = tot_t = tot_kl = 0.0
    n = 0
    try:
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = to_device(batch, device)
            full = build_masks(batch["valid"], batch["prefix_len"], False)
            blocked = build_masks(batch["valid"], batch["prefix_len"], True)
            t_logits = logits_of(teacher, batch, full)
            s_logits = logits_of(student, batch, blocked)
            tot_t += float(weighted_ce(t_logits, batch["labels"], w))
            tot_s += float(weighted_ce(s_logits, batch["labels"], w))
            tot_kl += float(weighted_kd(s_logits, t_logits, batch["labels"], w))
            n += 1
    finally:
        random.setstate(py)
        torch.set_rng_state(th)
        student.train()
    d = max(n, 1)
    return tot_s / d, tot_t / d, tot_kl / d


def lr_at(step, steps, lr, warmup=0.03, min_ratio=0.1):
    w = max(int(warmup * steps), 1)
    if step < w:
        return lr * (step + 1) / w
    p = (step - w) / max(steps - w, 1)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default="k2-fsa/OmniVoice")
    p.add_argument("--init-from", default=None,
                   help="student init (default: the teacher, isolating the mask change)")
    p.add_argument("--train-manifest", default="data/dataset_without_dev.csv")
    p.add_argument("--dev-manifest", default="data/dev_set.csv")
    p.add_argument("--dev-ref-manifest", default=None,
                   help="where dev references are drawn from (default: the train "
                        "manifest). Dev targets stay held out; only the reference "
                        "voice comes from the training split.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-speaker-holdout", dest="speaker_holdout",
                   action="store_false",
                   help="keep dev speakers in training (the old behaviour): "
                        "validation then measures utterance generalisation only")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    p.add_argument("--teacher-dtype", default="bf16", choices=list(DTYPES))
    p.add_argument("--ref-cap", type=int, default=150, help="reference frames (25/s)")
    p.add_argument("--max-frames", type=int, default=750)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.0,
                   help="ground-truth CE, added to the KD term. CE is ~40x "
                        "larger than KD here, so useful values are small; "
                        "see probe_loss_balance.py")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    os.makedirs(args.out_dir, exist_ok=True)

    tpath = _resolve_model_path(args.teacher)
    tok = AutoTokenizer.from_pretrained(tpath)
    teacher = OmniVoice.from_pretrained(
        tpath, train=True, dtype=DTYPES[args.teacher_dtype], attn_implementation="sdpa"
    ).to(device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    student = OmniVoice.from_pretrained(
        _resolve_model_path(args.init_from or args.teacher),
        train=True, dtype=DTYPES[args.dtype], attn_implementation="sdpa",
    ).to(device).train()
    if args.grad_checkpointing:
        student.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    print(f"device={device}  teacher={sum(x.numel() for x in teacher.parameters())/1e6:.1f}M"
          f"  student={sum(x.numel() for x in student.parameters())/1e6:.1f}M")

    C = student.config.num_audio_codebook
    w = torch.tensor(student.normalized_audio_codebook_weights, device=device)
    coll = CloneCollator(tok, C, student.config.audio_mask_id)

    def make(manifest, shuffle, workers, ref_manifest=None, **kw):
        ds = ClonePairDataset(manifest, args.data_root, ref_manifest=ref_manifest,
                              max_frames=args.max_frames, ref_cap=args.ref_cap, **kw)
        return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                              collate_fn=coll, num_workers=workers, drop_last=shuffle)

    # `dataset_without_dev.csv` removes the dev *utterances* but keeps their
    # speakers: every one of the 218 dev speakers still has other utterances in
    # train (median 13). That makes the default split utterance-held-out, which
    # cannot answer the question this model is judged on -- can it clone a voice
    # it has never heard? This model has no speaker embedding and no speaker ID
    # token, so a voice seen 13 times in training is exactly where timbre could
    # have been absorbed into the weights unnoticed.
    #
    # Holding out speakers moves those utterances from training into the
    # validation reference pool, so validation targets AND their reference clips
    # are both from voices the student never saw.
    val_speakers = None
    ref_rows = read_manifest(args.dev_ref_manifest or args.train_manifest)
    if args.speaker_holdout:
        val_speakers = {r["speaker_id"] for r in read_manifest(args.dev_manifest)}
        ref_rows = [r for r in ref_rows if r["speaker_id"] in val_speakers]

    train_ds, train_dl = make(args.train_manifest, True, args.num_workers,
                              exclude_speakers=val_speakers)
    dev_ds, dev_dl = make(args.dev_manifest, False, 0,
                          ref_manifest=read_manifest(args.dev_manifest) + ref_rows)
    print(f"train pairs {len(train_ds)}   val pairs {len(dev_ds)}")
    if args.speaker_holdout:
        full_n = len(read_manifest(args.train_manifest))
        print(f"speaker holdout: {len(val_speakers)} validation speakers, "
              f"{full_n - len(train_ds.rows)} training rows withheld "
              f"({(full_n - len(train_ds.rows))/full_n*100:.1f}%); "
              f"validation voices are unseen in training")
    else:
        print("WARNING: --no-speaker-holdout -- validation speakers also appear "
              "in training, so validation measures utterance generalisation only")
    if len(dev_ds) == 0:
        raise SystemExit("no validation target has a same-speaker reference available")

    params = [q for q in student.parameters() if q.requires_grad]
    decay = [q for q in params if q.ndim > 1]
    nodecay = [q for q in params if q.ndim <= 1]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    amp = device.type == "cuda"

    s0, t0, k0 = evaluate(student, teacher, dev_dl, device, w,
                          max_batches=args.eval_batches)
    print(f"before training: val KL {k0:.4f}   student(blocked) CE {s0:.4f}   "
          f"teacher(full) CE {t0:.4f}   gap {s0 - t0:+.4f}")

    step = micro = 0
    run = run_kd = run_ce = 0.0
    it = iter(train_dl)
    start = time.time()
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl)
            batch = next(it)
        batch = to_device(batch, device)
        full = build_masks(batch["valid"], batch["prefix_len"], False)
        blocked = build_masks(batch["valid"], batch["prefix_len"], True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            with torch.no_grad():
                t_logits = logits_of(teacher, batch, full)
            s_logits = logits_of(student, batch, blocked)
            kd = weighted_kd(s_logits, t_logits, batch["labels"], w, args.temperature)
            # Ground-truth CE anchors the student to the data, so it cannot inherit
            # teacher quirks wholesale. Secondary by design -- see --ce-weight.
            ce = (weighted_ce(s_logits, batch["labels"], w)
                  if args.ce_weight > 0 else torch.zeros((), device=device))
            loss = args.kd_weight * kd + args.ce_weight * ce

        (loss / args.grad_accum).backward()
        run += float(loss.detach())
        run_kd += float(kd.detach())
        run_ce += float(ce.detach())
        micro += 1
        if micro % args.grad_accum:
            continue

        for g in optim.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad(set_to_none=True)
        step += 1

        if args.log_every and step % args.log_every == 0:
            peak = (f"  peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB"
                    if device.type == "cuda" else "")
            d = args.log_every * args.grad_accum
            print(f"  step {step}/{args.steps}  loss {run/d:.4f}"
                  f"  kd {run_kd/d:.4f}  ce {run_ce/d:.4f}"
                  f"  lr {lr_at(step, args.steps, args.lr):.2e}"
                  f"  {step/max(time.time()-start,1e-6):.2f} step/s{peak}", flush=True)
            run = run_kd = run_ce = 0.0
        if args.eval_every and step % args.eval_every == 0:
            s, t, k = evaluate(student, teacher, dev_dl, device, w,
                               max_batches=args.eval_batches)
            print(f"  step {step}: val KL {k:.4f} (init {k0:.4f})  "
                  f"student(blocked) CE {s:.4f}  teacher(full) CE {t:.4f}"
                  f"  gap {s - t:+.4f}", flush=True)
        if args.save_every and step % args.save_every == 0:
            student.save_pretrained(os.path.join(args.out_dir, f"step_{step}"))

    s, t, k = evaluate(student, teacher, dev_dl, device, w,
                       max_batches=args.eval_batches)
    print(f"\nfinal: val KL {k:.4f} (was {k0:.4f})  student(blocked) CE {s:.4f}  "
          f"teacher(full) CE {t:.4f}  gap {s - t:+.4f} (was {s0 - t0:+.4f})")
    student.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"saved {args.out_dir}")


if __name__ == "__main__":
    main()
