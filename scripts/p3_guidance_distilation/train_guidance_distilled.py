#!/usr/bin/env python3
"""Distil classifier-free guidance into a single conditional pass.

Inference currently runs the model TWICE per diffusion step: a conditional pass
over `[style | text | ref_audio | target]` and an unconditional pass over the
target alone, mixed as

    log p = log_softmax( log p_c + w * (log p_c - log p_u) )

(`_predict_tokens_with_scoring`). Measured forward-time ratio for B=2 vs B=1 is
1.91-2.04x, so CFG costs a true ~2x -- the unconditional branch is padded to the
conditional length, so its compute happens over the full sequence either way.

This trains the student's **conditional pass alone** to reproduce that mixed
distribution. Afterwards `guidance_scale=0` uses one branch and gets the guided
result, halving inference cost.

The teacher runs both branches under `torch.no_grad`; the student runs only the
conditional one. Two loss terms, mirroring `train_prefix_blocked.py`:

  KD  KL(CFG-mixed teacher || student), codebook-weighted, masked positions only
  CE  ground-truth cross-entropy, off by default -- see --ce-weight

Composing with stage 2: point BOTH --teacher and --model at your prefix-blocked
checkpoint and pass --prefix-blocked. The teacher then differs from the student
only by having CFG, which isolates guidance removal from every other change.

    # continue from a prefix-blocked checkpoint, isolating the CFG change
    python scripts/p3_guidance_distilation/train_guidance_distilled.py \\
        --teacher runs/prefix_blocked --model runs/prefix_blocked --prefix-blocked \\
        --out-dir runs/guidance --steps 10000 --lr 1e-5

    # from the stock teacher, with a ground-truth anchor
    python scripts/p3_guidance_distilation/train_guidance_distilled.py \\
        --out-dir runs/guidance --ce-weight 0.03

NOTE: this also removes auto mode. `drop_cond` and the CFG unconditional branch
are the same code path (`processor.py:150-154`), so a student that no longer needs
the unconditional pass can no longer generate without a reference.
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.dirname(_SCRIPTS))
# The stage directories carry a `pN_` ordering prefix that has been renumbered
# before, so locate the sibling by suffix rather than hardcoding the number.
_PB = next((os.path.join(_SCRIPTS, d) for d in sorted(os.listdir(_SCRIPTS))
            if d.endswith("prefix_blocking")), None)
if _PB is None:
    raise SystemExit("cannot find the prefix-blocking stage directory next to "
                     f"{_HERE} -- it provides the dataset and loss helpers")
sys.path.insert(0, _PB)

from transformers import AutoTokenizer  # noqa: E402

from omnivoice.models.omnivoice import (  # noqa: E402
    OmniVoice, _combine_text, _resolve_model_path, _tokenize_with_nonverbal_tags,
)
from train_prefix_blocked import (  # noqa: E402
    DTYPES, ClonePairDataset, build_masks, lr_at, read_manifest, to_device,
    weighted_ce,
)


# ---------------------------------------------------------------------------
# Data: the conditional sequence AND the unconditional (target-only) one
# ---------------------------------------------------------------------------


class GuidanceCollator:
    """Builds both CFG branches from one masked target.

    The unconditional branch must see the *same* masked target as the conditional
    one, or the two distributions are not comparable and the mix is meaningless.
    At inference it is built as `input_ids[..., -u_len:]` -- the target region
    alone, with no style, text or reference audio (`omnivoice.py:1342`), which is
    the same topology `drop_cond` produces during training.
    """

    def __init__(self, tokenizer, num_codebooks, mask_id, denoise=True):
        self.tok = tokenizer
        self.C = num_codebooks
        self.mask_id = mask_id
        self.denoise = denoise

    def _prefix_ids(self, s):
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
            ratio = random.uniform(0.0, 1.0)           # diffusion noise level
            tm = torch.rand(self.C, T) < ratio
            if tm.sum() == 0:
                tm[random.randrange(self.C), random.randrange(T)] = True
            inp = tgt.clone()
            inp[tm] = self.mask_id                     # shared by both branches
            lab = torch.full((self.C, T), -100, dtype=torch.long)
            lab[tm] = tgt[tm]
            built.append((torch.cat([pre, inp], dim=1), inp, lab,
                          pre.shape[1], audio_start, T))

        B = len(built)
        S = max(x[0].shape[1] for x in built)
        Tm = max(x[5] for x in built)

        ids = torch.full((B, self.C, S), self.mask_id, dtype=torch.long)
        audio_mask = torch.zeros(B, S, dtype=torch.bool)
        valid = torch.zeros(B, S, dtype=torch.bool)
        prefix_len = torch.zeros(B, dtype=torch.long)

        u_ids = torch.full((B, self.C, Tm), self.mask_id, dtype=torch.long)
        u_valid = torch.zeros(B, Tm, dtype=torch.bool)

        labels = torch.full((B, self.C, Tm), -100, dtype=torch.long)
        # Where each target position lives in the conditional sequence, so the two
        # branches' logits can be lined up despite per-item prefix lengths.
        tgt_index = torch.zeros(B, Tm, dtype=torch.long)
        tgt_valid = torch.zeros(B, Tm, dtype=torch.bool)

        for b, (x, u, lab, P, audio_start, T) in enumerate(built):
            L = x.shape[1]
            ids[b, :, :L] = x
            audio_mask[b, audio_start:L] = True
            valid[b, :L] = True
            prefix_len[b] = P

            u_ids[b, :, :T] = u
            u_valid[b, :T] = True

            labels[b, :, :T] = lab
            tgt_index[b, :T] = torch.arange(P, P + T)
            tgt_valid[b, :T] = True

        return {
            "input_ids": ids, "audio_mask": audio_mask, "valid": valid,
            "prefix_len": prefix_len,
            "position_ids": torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
            "u_input_ids": u_ids,
            "u_audio_mask": u_valid.clone(),      # every uncond position is audio
            "u_valid": u_valid,
            "u_position_ids": torch.arange(Tm).unsqueeze(0).expand(B, Tm).contiguous(),
            "labels": labels, "tgt_index": tgt_index, "tgt_valid": tgt_valid,
        }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def logits_of(model, ids, audio_mask, attn, position_ids):
    e = model._prepare_embed_inputs(ids, audio_mask)
    h = model.llm(inputs_embeds=e, attention_mask=attn,
                  position_ids=position_ids, return_dict=True).last_hidden_state
    b, s, _ = h.shape
    return model.audio_heads(h).view(
        b, s, model.config.num_audio_codebook, model.config.audio_vocab_size
    ).permute(0, 2, 1, 3)


def gather_target(logits, tgt_index):
    """[B, C, S, V] -> [B, C, T, V], picking each item's target positions."""
    B, C, _, V = logits.shape
    idx = tgt_index[:, None, :, None].expand(B, C, tgt_index.shape[1], V)
    return logits.gather(2, idx)


def cfg_log_probs(c_logits, u_logits, guidance_scale, mask_id):
    """The distribution the sampler actually commits from.

    Reproduces `_predict_tokens_with_scoring`, including the suppression of
    `audio_mask_id` *after* mixing -- the student should learn the distribution
    that is sampled, not an unnormalised precursor to it.
    """
    c = torch.log_softmax(c_logits, dim=-1)
    if guidance_scale == 0:
        lp = c
    else:
        u = torch.log_softmax(u_logits, dim=-1)
        lp = torch.log_softmax(c + guidance_scale * (c - u), dim=-1)
    lp = lp.clone()
    lp[..., mask_id] = -float("inf")
    return lp


def weighted_kd_logp(s_logits, t_logp, labels, w, temperature=1.0, topk=0):
    """KL(teacher || student), teacher given as log-probabilities.

    `t_logp` contains -inf at the suppressed mask id, and `0 * -inf` is NaN, so
    zero-probability entries are dropped explicitly rather than multiplied through.

    With ``topk > 0`` this becomes a **grouped** KL: the teacher's top-k outcomes
    are kept individually and everything else is lumped into a single "rest"
    bucket. The motivation is that the gradient of a full KL is `p_s - p_t` per
    entry, which is NOT probability-weighted -- measured on this model, ~37% of
    the gradient magnitude lands outside the teacher's top-64, on entries the
    sampler never reads (it consumes only `argmax` and `max log p`).

    The rest bucket is what makes the coarsening sound rather than merely cheaper.
    Without it the student could pile mass onto a token outside the teacher's
    top-k for free -- and that token could become its argmax, which is precisely
    what the sampler does read. Lumping penalises any mass leaving the head while
    leaving the tail's internal arrangement unconstrained. Formally this is a KL
    on a coarsened partition, so by the data-processing inequality it lower-bounds
    the full KL.
    """
    t = temperature
    lps = torch.log_softmax(s_logits / t, dim=-1)

    if topk and topk < t_logp.shape[-1]:
        t_top, idx = t_logp.topk(topk, dim=-1)
        s_top = lps.gather(-1, idx)
        pt = t_top.exp()
        term = torch.where(pt > 0, pt * (t_top - s_top), torch.zeros_like(pt))
        kl = term.sum(dim=-1)
        # The lumped remainder. Clamped because both sums approach 1 and the
        # complement is a small difference of large numbers in fp32.
        t_rest = (1.0 - pt.sum(dim=-1)).clamp_min(1e-9)
        s_rest = (1.0 - s_top.exp().sum(dim=-1)).clamp_min(1e-9)
        kl = kl + t_rest * (t_rest.log() - s_rest.log())
    else:
        pt = t_logp.exp()
        term = torch.where(pt > 0, pt * (t_logp - lps), torch.zeros_like(pt))
        kl = term.sum(dim=-1)                               # [B, C, T]

    mask = (labels != -100).float()
    per_cb = (kl * mask).sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp(min=1.0)
    return (per_cb * w).sum() * t * t


def forward_pair(teacher, student, batch, args, mask_id):
    """Teacher CFG-mixed target and the student's conditional logits, aligned."""
    cond_attn = build_masks(batch["valid"], batch["prefix_len"], args.prefix_blocked)
    u_attn = (batch["u_valid"][:, None, None, :]
              .expand(-1, 1, batch["u_valid"].shape[1], -1).contiguous())

    with torch.no_grad():
        t_c = gather_target(
            logits_of(teacher, batch["input_ids"], batch["audio_mask"],
                      cond_attn, batch["position_ids"]), batch["tgt_index"])
        t_u = logits_of(teacher, batch["u_input_ids"], batch["u_audio_mask"],
                        u_attn, batch["u_position_ids"])
        t_logp = cfg_log_probs(t_c, t_u, args.guidance_scale, mask_id)

    s = gather_target(
        logits_of(student, batch["input_ids"], batch["audio_mask"],
                  cond_attn, batch["position_ids"]), batch["tgt_index"])
    return s, t_logp, t_c


@torch.no_grad()
def evaluate(student, teacher, loader, device, w, args, mask_id,
             seed=1234, max_batches=0):
    """Held-out KL to the guided teacher, plus CE for both.

    Also reports the *unguided* baseline: KL between the teacher's own conditional
    pass and its CFG-mixed output. That is the gap guidance distillation has to
    close, so a student KL below it means training has bought something.
    """
    py, th = random.getstate(), torch.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)
    student.eval()
    tot = {"kl": 0.0, "ce_s": 0.0, "ce_t": 0.0, "kl_uncond": 0.0}
    n = 0
    try:
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = to_device(batch, device)
            s, t_logp, t_c = forward_pair(teacher, student, batch, args, mask_id)
            # topk is deliberately NOT passed here: the reported number must stay
            # the full 1025-way KL, or it cannot be compared against other runs.
            tot["kl"] += float(weighted_kd_logp(s, t_logp, batch["labels"], w))
            tot["kl_uncond"] += float(
                weighted_kd_logp(t_c, t_logp, batch["labels"], w))
            tot["ce_s"] += float(weighted_ce(s, batch["labels"], w))
            tot["ce_t"] += float(weighted_ce(t_c, batch["labels"], w))
            n += 1
    finally:
        random.setstate(py)
        torch.set_rng_state(th)
        student.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default="k2-fsa/OmniVoice",
                   help="runs both CFG branches; point at your stage-2 checkpoint "
                        "to isolate guidance removal")
    p.add_argument("--model", "--init-from", dest="model", default=None,
                   help="student init (default: the teacher). Use your "
                        "prefix-blocked checkpoint to continue from stage 2.")
    p.add_argument("--train-manifest", default="data/dataset_without_dev.csv")
    p.add_argument("--dev-manifest", default="data/dev_set.csv")
    p.add_argument("--dev-ref-manifest", default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-speaker-holdout", dest="speaker_holdout",
                   action="store_false",
                   help="keep dev speakers in training; validation then measures "
                        "utterance generalisation only")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix-blocked", action="store_true",
                   help="run the conditional branch with prefix->target attention "
                        "cut, for continuing from a stage-2 checkpoint")
    p.add_argument("--guidance-scale", type=float, default=2.0,
                   help="the w the student learns to bake in (inference default 2.0)")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    p.add_argument("--teacher-dtype", default="bf16", choices=list(DTYPES))
    p.add_argument("--ref-cap", type=int, default=150)
    p.add_argument("--max-frames", type=int, default=750)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kd-topk", type=int, default=0, metavar="K",
                   help="grouped KL over the teacher's top-K plus one lumped "
                        "'rest' bucket (0 = full 1025-way KL). Validation always "
                        "reports the FULL KL so runs stay comparable.")
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.0,
                   help="ground-truth CE added to the KD term. CE is far larger "
                        "than KD here, so useful values are small -- run "
                        "scripts/prefix_blocking/probe_loss_balance.py to pick one")
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
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)

    tpath = _resolve_model_path(args.teacher)
    tok = AutoTokenizer.from_pretrained(tpath)
    teacher = OmniVoice.from_pretrained(
        tpath, train=True, dtype=DTYPES[args.teacher_dtype],
        attn_implementation="sdpa").to(device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    student = OmniVoice.from_pretrained(
        _resolve_model_path(args.model or args.teacher),
        train=True, dtype=DTYPES[args.dtype],
        attn_implementation="sdpa").to(device).train()
    if args.grad_checkpointing:
        student.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    mask_id = student.config.audio_mask_id
    print(f"device={device}  teacher={args.teacher}  student={args.model or args.teacher}")
    print(f"guidance_scale={args.guidance_scale}  prefix_blocked={args.prefix_blocked}"
          f"  kd_weight={args.kd_weight}  ce_weight={args.ce_weight}")

    C = student.config.num_audio_codebook
    w = torch.tensor(student.normalized_audio_codebook_weights, device=device)
    coll = GuidanceCollator(tok, C, mask_id)

    def make(manifest, shuffle, workers, ref_manifest=None, **kw):
        ds = ClonePairDataset(manifest, args.data_root, ref_manifest=ref_manifest,
                              max_frames=args.max_frames, ref_cap=args.ref_cap, **kw)
        return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                              collate_fn=coll, num_workers=workers, drop_last=shuffle)

    val_speakers = None
    ref_rows = read_manifest(args.dev_ref_manifest or args.train_manifest)
    if args.speaker_holdout:
        val_speakers = {r["speaker_id"] for r in read_manifest(args.dev_manifest)}
        ref_rows = [r for r in ref_rows if r["speaker_id"] in val_speakers]

    train_ds, train_dl = make(args.train_manifest, True, args.num_workers,
                              exclude_speakers=val_speakers)
    dev_ds, dev_dl = make(args.dev_manifest, False, 0,
                          ref_manifest=read_manifest(args.dev_manifest) + ref_rows)
    print(f"train pairs {len(train_ds)}   val pairs {len(dev_ds)}"
          f"{'  (validation voices unseen in training)' if args.speaker_holdout else ''}")
    if len(dev_ds) == 0:
        raise SystemExit("no validation target has a same-speaker reference available")

    params = [q for q in student.parameters() if q.requires_grad]
    optim = torch.optim.AdamW(
        [{"params": [q for q in params if q.ndim > 1], "weight_decay": args.weight_decay},
         {"params": [q for q in params if q.ndim <= 1], "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    amp = device.type == "cuda"

    m0 = evaluate(student, teacher, dev_dl, device, w, args, mask_id,
                  max_batches=args.eval_batches)
    print(f"before training: val KL {m0['kl']:.4f}   "
          f"(teacher's own unguided pass: {m0['kl_uncond']:.4f})   "
          f"student CE {m0['ce_s']:.4f}   teacher CE {m0['ce_t']:.4f}")

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

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            s_logits, t_logp, _ = forward_pair(teacher, student, batch, args, mask_id)
            kd = weighted_kd_logp(s_logits, t_logp, batch["labels"], w,
                                  args.temperature, topk=args.kd_topk)
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
            print(f"  step {step}/{args.steps}  loss {run/d:.4f}  kd {run_kd/d:.4f}"
                  f"  ce {run_ce/d:.4f}  lr {lr_at(step, args.steps, args.lr):.2e}"
                  f"  {step/max(time.time()-start,1e-6):.2f} step/s{peak}", flush=True)
            run = run_kd = run_ce = 0.0
        if args.eval_every and step % args.eval_every == 0:
            m = evaluate(student, teacher, dev_dl, device, w, args, mask_id,
                         max_batches=args.eval_batches)
            print(f"  step {step}: val KL {m['kl']:.4f} (init {m0['kl']:.4f}, "
                  f"unguided {m['kl_uncond']:.4f})  student CE {m['ce_s']:.4f}"
                  f"  teacher CE {m['ce_t']:.4f}", flush=True)
        if args.save_every and step % args.save_every == 0:
            student.save_pretrained(os.path.join(args.out_dir, f"step_{step}"))

    m = evaluate(student, teacher, dev_dl, device, w, args, mask_id,
                 max_batches=args.eval_batches)
    print(f"\nfinal: val KL {m['kl']:.4f} (was {m0['kl']:.4f}; the teacher's own "
          f"unguided pass sits at {m['kl_uncond']:.4f})  "
          f"student CE {m['ce_s']:.4f}  teacher CE {m['ce_t']:.4f}")
    student.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"saved {args.out_dir}\n\nEvaluate with guidance_scale=0:\n"
          f"  python scripts/eval/generate_samples.py --model {args.out_dir} "
          f"--guidance-scale 0"
          f"{' --prefix-cached' if args.prefix_blocked else ''}")


if __name__ == "__main__":
    main()
