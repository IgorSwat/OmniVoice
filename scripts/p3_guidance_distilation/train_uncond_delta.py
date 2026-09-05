#!/usr/bin/env python3
"""Train an UncondDelta head so one forward pass yields the guided distribution.

`train_guidance_distilled.py` bakes guidance into `audio_heads`, which fixes the
scale at training time. This keeps the backbone FROZEN and trains a small head
that predicts the guidance direction

    Delta ~ log p_c - log p_u

from the conditional hidden states of the target region. Unconditional logits are
`c - Delta`, so the guided mixture is exactly `log_softmax(c + w * Delta)` and `w`
stays a runtime knob. One backbone forward at inference, ~4% extra for the head.

The model is its own teacher: the frozen backbone runs the real conditional and
unconditional branches under no_grad, and the single loss is

    KL( softmax(c + w (c_lp - u_lp))  ||  softmax(c + w * Delta) )

i.e. on the MIXTURE the sampler consumes, not on p_u -- errors in p_u enter the
mixture multiplied by w, so training on the branch understates them (measured:
a linear p_u head at KL 0.2 on the branch). Because `c` is shared and frozen,
matching the mixture at any nonzero w identifies Delta up to a per-position
constant the softmax removes; sampling w per batch (--guidance-range) constrains
it everywhere the mixture could put mass at any scale you might deploy.

The head is a 1-2 layer model of the backbone's own class (target-only attention,
initialised from the backbone's LAST layers) plus a zero-initialised projection,
so at step 0 Delta = 0 and guidance is a no-op.

    python scripts/p3_guidance_distilation/train_uncond_delta.py \\
        --model models/p4/round_07_tuned_with_kd --prefix-blocked \\
        --out-dir runs/uncond_delta --steps 4000 --lr 1e-4
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from train_guidance_distilled import (  # noqa: E402
    DTYPES, FRAME_RATE, AutoTokenizer, GuidanceCollator, LengthGroupedBatchSampler,
    OmniVoice, PromptSplitDataset, _resolve_model_path, build_masks, cfg_log_probs,
    gather_target, lr_at, prompt_split_lengths, read_manifest, row_stats, to_device,
    weighted_ce, weighted_kd_logp,
)


def install_delta_head(model, layers, ffn):
    """Give `model` a delta head initialised from its own last `layers` layers."""
    model.config.uncond_head = "delta"
    model.config.uncond_head_layers = layers
    model.config.uncond_head_ffn = ffn
    fresh = OmniVoice(model.config)
    fresh.load_state_dict(model.state_dict(), strict=False)
    src = fresh.llm.layers[-layers:]
    same_ffn = (ffn == 0 or ffn == model.config.llm_config.intermediate_size)
    for dst, s in zip(fresh.uncond_delta_llm.layers, src):
        sd = s.state_dict() if same_ffn else {k: v for k, v in s.state_dict().items() if "mlp" not in k}
        dst.load_state_dict(sd, strict=same_ffn)
    fresh.uncond_delta_llm.norm.load_state_dict(fresh.llm.norm.state_dict())
    # PreTrainedModel.post_init() re-initialises every Linear AFTER __init__'s
    # zeros_, so the projection must be zeroed here or Delta starts random and
    # the mixture KL at step 0 is ~19 instead of the Delta=0 gap.
    torch.nn.init.zeros_(fresh.uncond_delta.weight)
    return fresh


def cond_forward(model, batch, prefix_blocked):
    """Conditional branch: (target hidden states, target logits, valid, positions)."""
    attn = build_masks(batch["valid"], batch["prefix_len"], prefix_blocked)
    e = model._prepare_embed_inputs(batch["input_ids"], batch["audio_mask"])
    h = model.llm(inputs_embeds=e, attention_mask=attn,
                  position_ids=batch["position_ids"], return_dict=True).last_hidden_state
    B, S, H = h.shape
    c = model.audio_heads(h).view(B, S, model.config.num_audio_codebook,
                                  model.config.audio_vocab_size).permute(0, 2, 1, 3)
    idx = batch["tgt_index"]
    ht = h.gather(1, idx[:, :, None].expand(B, idx.shape[1], H))
    return ht, gather_target(c, idx), batch["tgt_valid"], idx


def uncond_forward(model, batch):
    u_attn = (batch["u_valid"][:, None, None, :]
              .expand(-1, 1, batch["u_valid"].shape[1], -1).contiguous())
    e = model._prepare_embed_inputs(batch["u_input_ids"], batch["u_audio_mask"])
    h = model.llm(inputs_embeds=e, attention_mask=u_attn,
                  position_ids=batch["u_position_ids"], return_dict=True).last_hidden_state
    B, S, _ = h.shape
    return model.audio_heads(h).view(B, S, model.config.num_audio_codebook,
                                     model.config.audio_vocab_size).permute(0, 2, 1, 3)


def step_losses(model, batch, w, args, mask_id):
    """Teacher mixture (no grad) and the head's mixture at the same w."""
    with torch.no_grad():
        ht, c_t, valid, pos = cond_forward(model, batch, args.prefix_blocked)
        u_t = uncond_forward(model, batch)
        t_logp = cfg_log_probs(c_t.float(), u_t.float(), w, mask_id)
    delta = model.uncond_delta_logits(ht, valid, pos).float()
    s_logits = c_t.float() + w * delta
    return s_logits, t_logp, c_t


@torch.no_grad()
def evaluate(model, loader, device, w, args, mask_id, cbw, seed=1234, max_batches=0):
    """Mixture KL at the fixed eval scale, plus the Delta=0 baseline (the gap)."""
    py, th = random.getstate(), torch.get_rng_state()
    random.seed(seed); torch.manual_seed(seed)
    tot = {"kl": 0.0, "kl_gap": 0.0, "ce": 0.0}; n = 0
    try:
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = to_device(batch, device)
            s, t, c = step_losses(model, batch, w, args, mask_id)
            tot["kl"] += float(weighted_kd_logp(s, t, batch["labels"], cbw))
            tot["kl_gap"] += float(weighted_kd_logp(c.float(), t, batch["labels"], cbw))
            tot["ce"] += float(weighted_ce(s, batch["labels"], cbw))
            n += 1
    finally:
        random.setstate(py); torch.set_rng_state(th)
    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="k2-fsa/OmniVoice",
                   help="backbone; it is frozen and acts as its own teacher")
    p.add_argument("--train-manifest", default="data/dataset_without_dev.csv")
    p.add_argument("--dev-manifest", default="data/dev_set.csv")
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-speaker-holdout", dest="speaker_holdout", action="store_false")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix-blocked", action="store_true",
                   help="stage-2 topology for the conditional branch; REQUIRED for "
                        "p2 descendants")
    p.add_argument("--head-layers", type=int, default=1)
    p.add_argument("--head-ffn", type=int, default=0,
                   help="head FFN width (0 = backbone's; a different width skips "
                        "the MLP init from the backbone)")
    p.add_argument("--guidance-scale", type=float, default=2.0,
                   help="eval scale, and the training scale unless --guidance-range")
    p.add_argument("--guidance-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="sample w ~ U(LO, HI) per batch so Delta is constrained at "
                        "every scale you might deploy")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    p.add_argument("--max-frames", type=int, default=750)
    p.add_argument("--min-target-seconds", type=float, default=2.0)
    p.add_argument("--max-prompt-ratio", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batch-tokens", type=int, default=16384)
    p.add_argument("--max-batch-size", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)

    mpath = _resolve_model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath)
    model = OmniVoice.from_pretrained(mpath, train=True, dtype=DTYPES[args.dtype],
                                      attn_implementation="sdpa")
    if model.uncond_delta_llm is None:
        model = install_delta_head(model, args.head_layers, args.head_ffn)
    model = model.to(device)
    for q in model.parameters():
        q.requires_grad_(False)
    head_params = list(model.uncond_delta_llm.parameters()) + list(model.uncond_delta.parameters())
    for q in head_params:
        q.requires_grad_(True)
    model.eval()                                   # backbone stays eval; the head has no dropout
    n_head = sum(q.numel() for q in head_params)
    print(f"device={device}  model={args.model}  prefix_blocked={args.prefix_blocked}")
    print(f"delta head: {args.head_layers} layer(s), {n_head/1e6:.1f}M trainable; backbone frozen")
    print(f"guidance: eval w={args.guidance_scale}"
          + (f", train w ~ U{tuple(args.guidance_range)}" if args.guidance_range else ""))

    mask_id = model.config.audio_mask_id
    C = model.config.num_audio_codebook
    cbw = torch.tensor(model.normalized_audio_codebook_weights, device=device)
    coll = GuidanceCollator(tok, C, mask_id)
    min_tgt = int(round(args.min_target_seconds * FRAME_RATE))

    def make(manifest, shuffle, workers, **kw):
        ds = PromptSplitDataset(manifest, args.data_root, max_frames=args.max_frames,
                                min_target_frames=min_tgt,
                                max_prompt_ratio=args.max_prompt_ratio, **kw)
        if args.batch_tokens > 0:
            lengths = prompt_split_lengths(ds, row_stats(manifest, args.data_root, tok),
                                           args.max_frames)
            bs = LengthGroupedBatchSampler(lengths, args.batch_tokens,
                                           max_batch_size=args.max_batch_size,
                                           shuffle=shuffle, seed=args.seed)
            return ds, DataLoader(ds, batch_sampler=bs, collate_fn=coll,
                                  num_workers=workers), bs
        return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                              collate_fn=coll, num_workers=workers, drop_last=shuffle), None

    val_speakers = ({r["speaker_id"] for r in read_manifest(args.dev_manifest)}
                    if args.speaker_holdout else None)
    train_ds, train_dl, train_bs = make(args.train_manifest, True, args.num_workers,
                                        exclude_speakers=val_speakers)
    dev_ds, dev_dl, _ = make(args.dev_manifest, False, 0)
    print(f"train pairs {len(train_ds)}   val pairs {len(dev_ds)}")

    optim = torch.optim.AdamW(
        [{"params": [q for q in head_params if q.ndim > 1], "weight_decay": args.weight_decay},
         {"params": [q for q in head_params if q.ndim <= 1], "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    amp = device.type == "cuda"

    m0 = evaluate(model, dev_dl, device, args.guidance_scale, args, mask_id, cbw,
                  max_batches=args.eval_batches)
    print(f"before training: val mixture KL {m0['kl']:.4f}   (Delta=0 gap {m0['kl_gap']:.4f})")

    step = micro = 0; run = 0.0; it = iter(train_dl); start = time.time()
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            if train_bs is not None:
                train_bs.set_epoch(train_bs.epoch + 1)
            it = iter(train_dl); batch = next(it)
        batch = to_device(batch, device)
        w = (random.uniform(*args.guidance_range) if args.guidance_range
             else args.guidance_scale)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            s, t, _ = step_losses(model, batch, w, args, mask_id)
            loss = weighted_kd_logp(s, t, batch["labels"], cbw)
        (loss / args.grad_accum).backward()
        run += float(loss.detach()); micro += 1
        if micro % args.grad_accum:
            continue
        for g in optim.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(head_params, args.max_grad_norm)
        optim.step(); optim.zero_grad(set_to_none=True); step += 1

        if args.log_every and step % args.log_every == 0:
            d = args.log_every * args.grad_accum
            print(f"  step {step}/{args.steps}  mixture KL {run/d:.4f}  "
                  f"lr {lr_at(step, args.steps, args.lr):.2e}  "
                  f"{step/max(time.time()-start,1e-6):.2f} step/s", flush=True)
            run = 0.0
        if args.eval_every and step % args.eval_every == 0:
            m = evaluate(model, dev_dl, device, args.guidance_scale, args, mask_id, cbw,
                         max_batches=args.eval_batches)
            print(f"  step {step}: val mixture KL {m['kl']:.4f} (init {m0['kl']:.4f}, "
                  f"gap {m['kl_gap']:.4f})  CE {m['ce']:.4f}", flush=True)
        if args.save_every and step % args.save_every == 0:
            model.save_pretrained(os.path.join(args.out_dir, f"step_{step}"))

    m = evaluate(model, dev_dl, device, args.guidance_scale, args, mask_id, cbw,
                 max_batches=args.eval_batches)
    print(f"\nfinal: val mixture KL {m['kl']:.4f} (was {m0['kl']:.4f}; Delta=0 gap "
          f"{m['kl_gap']:.4f})  CE {m['ce']:.4f}")
    model.save_pretrained(args.out_dir); tok.save_pretrained(args.out_dir)
    print(f"saved {args.out_dir}\n\nGenerate with guidance from ONE pass:\n"
          f"  python scripts/eval/generate_samples.py --model {args.out_dir} "
          f"--guidance-scale {args.guidance_scale}"
          f"{' --prefix-blocked' if args.prefix_blocked else ''}")


if __name__ == "__main__":
    main()
