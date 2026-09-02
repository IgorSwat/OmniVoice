#!/usr/bin/env python3
"""Greedy layer-by-layer depth pruning with KD+CE recovery between every cut.

One layer per round. Each round:

  1. Score every remaining layer by the KD damage it causes when bypassed --
     measured, not predicted. Selection uses KD rather than CE because KD is far
     more sensitive here: dropping one layer moves CE by 0.3% but KD by 5.5x
     (measured 0.006 -> 0.034 on this checkpoint), so CE cannot resolve between
     candidates that KD separates cleanly.
  2. Remove the cheapest layer for real (not masked) and renumber the stack.
  3. Recover with KD **and** ground-truth CE against the ORIGINAL teacher.
  4. Gate: stop if the cheapest available cut already costs too much.

Why one at a time. Single-layer importance measured on the intact model is only
valid for the first cut -- once a layer is gone its neighbours absorb its
function. Measured on this checkpoint, layers 5-8 cost +0.323 CE individually but
+0.423 removed together, ~30% worse than additive, and the gap grows with the
number of simultaneous cuts. Re-scoring every round keeps the greedy choice
honest. It also keeps the two loss terms complementary: cos(g_kd, g_ce) is 0.175
after one cut but 0.950 after eight, i.e. CE stops adding independent information
once the model is badly damaged.

Why the teacher never moves. Each round distils against the original unpruned
model, not the previous round's output. A moving target means round n only has to
match an already-degraded model, and the degradation compounds silently with
nothing measuring the total.

Data follows the ORIGINAL training scheme, not the paired-reference layout used
in stages 2-3: the prompt is a prefix of the same utterance and the remainder is
the target (`OmniVoiceSampleProcessor`). The split is controlled per sample --
see `PromptSplitDataset`.

    python scripts/p4_width_pruning/progressive_depth_prune.py \\
        --model models/p2 --out-dir runs/depth --n-layers 8 --device cuda \\
        --batch-tokens 32768 --grad-checkpointing
"""

import argparse
import copy
import json
import os
import random
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from torch.utils.data import Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from omnivoice.models.omnivoice import OmniVoice, _resolve_model_path  # noqa: E402
from p4_width_pruning import distill, surgery  # noqa: E402
from p4_width_pruning.calibration import _audio_logits, _hidden_states  # noqa: E402
from p4_width_pruning.manifest import (  # noqa: E402
    CodecManifestDataset, build_dataloader, build_processor, seed_everything,
)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
FRAME_RATE = 25


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class PromptSplitDataset(Dataset):
    """Same-utterance prompt/target split with an explicit, bounded prompt length.

    The stock processor draws ``prompt_ratio ~ U(*prompt_ratio_range)`` and takes
    ``int(T * prompt_ratio)`` frames as the prompt, which on a short utterance can
    leave a target too small to be a meaningful training signal. Here the prompt is

        prompt = min( int(T * U(0, max_ratio)),  T - min_target_frames )

    so the target keeps at least ``--min-target-seconds`` and the prompt never
    exceeds ``--max-prompt-ratio`` of the utterance. The floor binds only on short
    utterances: at a 0.3 cap it takes effect below T = 50/0.7 ~= 71 frames (2.9 s).

    Injecting ``clean_start_token_idx`` is the processor's own hook for an explicit
    split (`processor.py:133`). It also prepends ``<|denoise|>`` to the style
    prefix (`processor.py:102`) -- which is what inference does whenever a
    reference is supplied, so this moves training *towards* the inference topology
    rather than away from it.
    """

    def __init__(self, base, min_target_frames, max_prompt_ratio):
        self.base = base
        self.min_target_frames = min_target_frames
        self.max_prompt_ratio = max_prompt_ratio

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        T = s["audio_tokens"].shape[1]
        cap = max(0, T - self.min_target_frames)
        s["label"] = dict(s["label"])
        s["label"]["clean_start_token_idx"] = min(
            int(T * random.uniform(0.0, self.max_prompt_ratio)), cap)
        return s

    # forwarded so the length-grouped sampler still works
    @property
    def lengths(self):
        return self.base.lengths

    @property
    def hours(self):
        return self.base.hours


# ---------------------------------------------------------------------------
# Prefix blocking
# ---------------------------------------------------------------------------


def block_prefix_(attention_mask, prefix_len):
    """Cut prefix-query -> target-key attention, in place.

    Same edit as stage 2: the target still sees the whole prefix, only the reverse
    direction is removed. Applied once per batch on the collator's freshly built
    all-ones mask, so reading `[:, 0, 0, :]` for the valid set is safe here (it
    would not be on a mask this has already touched).
    """
    B, _, L, _ = attention_mask.shape
    pos = torch.arange(L, device=attention_mask.device)
    valid = attention_mask[:, 0, 0, :].clone()
    is_prefix = pos[None, :] < prefix_len[:, None].to(attention_mask.device)
    is_target = valid & ~is_prefix
    attention_mask &= ~(is_prefix[:, None, :, None] & is_target[:, None, None, :])
    return attention_mask


def build_loader(ds, processor, batch_tokens, max_batch_size, shuffle, seed,
                 num_workers, prefix_blocked):
    """`build_dataloader`, plus the prefix block and the boundary it needs.

    The boundary is `audio_start + prompt_length`: `audio_start` is the first True
    in `audio_mask` (style and text precede it), and `prompt_length` is the value
    `PromptSplitDataset` injected as `clean_start_token_idx`. Blocking inside the
    collate function means every consumer downstream -- the training loop, the CE
    eval, the KD scorer -- sees the deployment topology without needing to know
    about it.
    """
    from torch.utils.data import DataLoader
    from omnivoice.data.collator import PaddingDataCollator
    from p4_width_pruning.manifest import LengthGroupedBatchSampler, _seed_worker

    collator = PaddingDataCollator(processor, batch_tokens)

    def collate(samples):
        processed, prompts = [], []
        for s in samples:
            processed.append(processor(s))
            prompts.append(s["label"].get("clean_start_token_idx", 0))
        batch = collator(processed)
        if prefix_blocked:
            audio_start = batch["audio_mask"].float().argmax(dim=1)
            prefix_len = audio_start + torch.tensor(prompts, dtype=torch.long)
            block_prefix_(batch["attention_mask"], prefix_len)
            batch["prefix_len"] = prefix_len
        return batch

    sampler = LengthGroupedBatchSampler(
        ds.lengths, batch_tokens=batch_tokens, max_batch_size=max_batch_size,
        shuffle=shuffle, seed=seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                        num_workers=num_workers,
                        worker_init_fn=_seed_worker if num_workers else None)
    loader.batch_sampler_ref = sampler
    return loader


# ---------------------------------------------------------------------------
# Surgery
# ---------------------------------------------------------------------------


def bypass(model, idx):
    """Temporarily make a layer the identity on the residual stream (reversible)."""
    lyr = model.llm.layers[idx]
    original = lyr.forward
    # Qwen3DecoderLayer.forward returns a bare tensor in transformers 5.x and the
    # next layer's norm consumes it directly; returning a tuple fails downstream.
    lyr.forward = lambda hidden_states, *a, **kw: hidden_states
    return lambda: setattr(lyr, "forward", original)


def remove_layer(model, idx):
    """Delete a layer for real and renumber the stack.

    ``layer_idx`` is baked into each attention module and indexes the KV cache, so
    leaving stale indices behind produces a model that trains but breaks the first
    time anything asks for a cache.
    """
    lm = model.llm
    lm.layers = torch.nn.ModuleList(
        [l for i, l in enumerate(lm.layers) if i != idx])
    for i, l in enumerate(lm.layers):
        l.self_attn.layer_idx = i
        if hasattr(l, "layer_idx"):
            l.layer_idx = i
    n = len(lm.layers)
    for cfg in (lm.config, model.config.llm_config):
        cfg.num_hidden_layers = n
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = list(cfg.layer_types)[:n]
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_kd(student, teacher, loader, device, w, max_batches=0, seed=1234):
    """Codebook-weighted KL(teacher || student) on held-out data.

    The RNG is pinned and restored for the same reason ``distill.evaluate_loss``
    does it: ``OmniVoiceSampleProcessor`` redraws ``mask_ratio ~ U(0,1)`` from the
    global ``random`` module on every call, and without pinning each candidate
    layer would be scored at a different noise level -- worth up to ~0.19 nats,
    far more than the differences being ranked.
    """
    py, th = random.getstate(), torch.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)
    was_training = student.training
    student.eval()
    total = 0.0
    n = 0
    try:
        for b, batch in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _, tl = _hidden_states(teacher, batch, output_hidden_states=False)
            _, sl = _hidden_states(student, batch, output_hidden_states=False)
            total += float(distill._weighted_kd(
                _audio_logits(student, sl), _audio_logits(teacher, tl).float(),
                batch["labels"], w))
            n += 1
    finally:
        random.setstate(py)
        torch.set_rng_state(th)
        if was_training:
            student.train()
    return total / max(n, 1)


def score_layers(student, teacher, loader, device, w, max_batches, protect):
    """KD damage from bypassing each remaining layer, cheapest first."""
    out = []
    for i in range(len(student.llm.layers)):
        if i in protect:
            continue
        undo = bypass(student, i)
        out.append((i, evaluate_kd(student, teacher, loader, device, w, max_batches)))
        undo()
    out.sort(key=lambda x: x[1])
    return out


# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("model / data")
    g.add_argument("--model", default="k2-fsa/OmniVoice")
    g.add_argument("--teacher", default=None,
                   help="KD target, fixed for the whole ladder (default: --model "
                        "as loaded before any cut)")
    g.add_argument("--train-manifest", default="data/dataset_without_dev.csv")
    g.add_argument("--dev-manifest", default="data/dev_set.csv")
    g.add_argument("--data-root", default="data")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--device", default=None)
    g.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    g.add_argument("--teacher-dtype", default="bf16", choices=list(DTYPES))
    g.add_argument("--attn", default="sdpa")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--min-frames", type=int, default=50)
    g.add_argument("--max-frames", type=int, default=2000)

    g = p.add_argument_group("prompt split")
    g.add_argument("--min-target-seconds", type=float, default=2.0,
                   help="hard floor on the target region; binds only on short "
                        "utterances (below ~2.9 s at the default 0.3 cap)")
    g.add_argument("--max-prompt-ratio", type=float, default=0.3)
    g.add_argument("--prefix-blocked", action="store_true",
                   help="train and evaluate with prefix->target attention cut. "
                        "REQUIRED when starting from a stage-2 checkpoint: that "
                        "model is deployed with the block, so recovering it under "
                        "full attention would spend every epoch pulling it back "
                        "towards a topology it is never used in.")

    g = p.add_argument_group("ladder")
    g.add_argument("--n-layers", type=int, default=8, help="how many to remove")
    g.add_argument("--protect", type=int, nargs="*", default=[0],
                   help="layer indices never cut, in ORIGINAL numbering for the "
                        "first round only. Layer 0 costs +3.93 CE to drop against "
                        "a +0.02 median, so it is excluded by default.")
    g.add_argument("--max-kd-damage", type=float, default=0.5,
                   help="stop if even the cheapest remaining cut exceeds this KD "
                        "increase over the round's starting point")
    g.add_argument("--select-batches", type=int, default=4,
                   help="dev batches per candidate when scoring. Every remaining "
                        "layer is scored each round, so this is the dominant "
                        "non-training cost: ~L x this many forwards per round.")
    g.add_argument("--dev-batches", type=int, default=0)

    g = p.add_argument_group("recovery")
    g.add_argument("--steps", type=int, default=0,
                   help="optimizer steps per round (0 = one epoch of the train set)")
    g.add_argument("--epochs", type=float, default=1.0,
                   help="used when --steps is 0")
    g.add_argument("--final-steps", type=int, default=0,
                   help="steps for the recovery after the LAST cut (0 = use "
                        "--final-epochs). The intermediate rounds only need to "
                        "make the next selection honest; the final model is the "
                        "one you keep, so it is usually worth training longer.")
    g.add_argument("--final-epochs", type=float, default=None,
                   help="epochs for the last recovery (default: same as --epochs)")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--batch-tokens", type=int, default=16384)
    g.add_argument("--max-batch-size", type=int, default=128)
    g.add_argument("--grad-accum", type=int, default=1)
    g.add_argument("--kd-weight", type=float, default=1.0)
    g.add_argument("--ce-weight", type=float, default=0.05,
                   help="ground-truth CE alongside KD. 0.05 puts CE at ~30%% of "
                        "the gradient after a single-layer cut (measured); it is "
                        "~1.7x the value appropriate for an intact model because "
                        "|g_kd| jumps ~7x on a cut while |g_ce| barely moves. "
                        "Re-probe every few rounds -- the balance drifts with damage.")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--grad-checkpointing", action="store_true")
    args = p.parse_args(argv)

    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = DTYPES[args.dtype]

    path = _resolve_model_path(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    teacher = OmniVoice.from_pretrained(
        _resolve_model_path(args.teacher or args.model), train=True,
        dtype=DTYPES[args.teacher_dtype], attn_implementation=args.attn).to(device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    student = OmniVoice.from_pretrained(
        path, train=True, dtype=dtype, attn_implementation=args.attn).to(device)
    if args.grad_checkpointing:
        student.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    w = torch.tensor(student.normalized_audio_codebook_weights, device=device)
    proc = build_processor(student.config, tok)
    ds_kw = {"tokenizer": tok, "min_frames": args.min_frames,
             "max_frames": args.max_frames}
    min_tgt = int(round(args.min_target_seconds * FRAME_RATE))
    wrap = lambda d: PromptSplitDataset(d, min_tgt, args.max_prompt_ratio)  # noqa: E731
    train_ds = wrap(CodecManifestDataset(args.train_manifest, args.data_root, **ds_kw))
    dev_ds = wrap(CodecManifestDataset(args.dev_manifest, args.data_root, **ds_kw))

    mk = lambda ds, bt, sh, nw: build_loader(  # noqa: E731
        ds, proc, bt, args.max_batch_size, sh, args.seed, nw, args.prefix_blocked)
    train_loader = mk(train_ds, args.batch_tokens, True, args.num_workers)
    # num_workers=0: the eval helpers pin the RNG, and worker processes draw
    # their masks outside that seeding.
    dev_loader = mk(dev_ds, args.batch_tokens, False, 0)

    per_epoch = max(1, len(train_loader) // args.grad_accum)
    steps = args.steps or max(1, int(per_epoch * args.epochs))
    final_epochs = args.epochs if args.final_epochs is None else args.final_epochs
    final_steps = args.final_steps or max(1, int(per_epoch * final_epochs))
    L0 = len(student.llm.layers)
    print(f"device={device}  layers={L0}  params={surgery.count_parameters(student)/1e6:.1f}M")
    print(f"train {len(train_ds)} utts ({train_ds.hours:.1f} h), "
          f"{per_epoch} steps/epoch -> {steps} per round ({args.epochs:g} ep)"
          + (f", {final_steps} for the final round "
             f"({final_steps/per_epoch:g} ep)" if final_steps != steps else ""))
    print(f"prompt split: <= {args.max_prompt_ratio:g} of the utterance, "
          f"target floor {args.min_target_seconds:g}s ({min_tgt} frames)")
    print(f"attention: {'PREFIX-BLOCKED (matches stage-2 deployment)' if args.prefix_blocked else 'full bidirectional'}")
    print(f"loss: kd {args.kd_weight} + ce {args.ce_weight}\n")

    teacher_ce = distill.evaluate_loss(teacher, dev_loader, device, args.dev_batches)
    report = {"args": vars(args), "teacher_ce": teacher_ce, "rounds": []}
    print(f"teacher dev CE {teacher_ce:.4f}\n")

    protect = set(args.protect)
    removed = []          # original indices, for the record
    alive = list(range(L0))

    for rnd in range(args.n_layers):
        t0 = time.time()
        base_kd = evaluate_kd(student, teacher, dev_loader, device, w, args.select_batches)
        print(f"=== round {rnd+1}/{args.n_layers}: {len(student.llm.layers)} layers, "
              f"KD {base_kd:.4f} ===")

        prot_now = {alive.index(i) for i in protect if i in alive}
        ranked = score_layers(student, teacher, dev_loader, device, w,
                              args.select_batches, prot_now)
        best, best_kd = ranked[0]
        damage = best_kd - base_kd
        top = "  ".join(f"L{alive[i]}:{v-base_kd:+.3f}" for i, v in ranked[:6])
        print(f"  cheapest candidates (original numbering)  {top}")

        if damage > args.max_kd_damage:
            print(f"\nstopping: cheapest cut costs {damage:+.4f} KD, over the "
                  f"--max-kd-damage {args.max_kd_damage} gate.")
            # The cut before this one was therefore the last, but it only got the
            # short recovery -- the ladder did not know it was ending. Give the
            # model the extended pass it would have had.
            if removed and final_steps > steps:
                print(f"running the extended final recovery anyway "
                      f"({final_steps} steps) on the {len(student.llm.layers)}-layer model")
                cfg = distill.DistillConfig(
                    steps=final_steps, lr=args.lr, grad_accum=args.grad_accum,
                    kd_weight=args.kd_weight, ce_weight=args.ce_weight,
                    hidden_weight=0.0, temperature=args.temperature,
                    log_every=args.log_every, eval_every=args.eval_every,
                    save_every=0, amp_dtype="bf16")
                fin_dir = os.path.join(args.out_dir, "final_recovery")
                os.makedirs(fin_dir, exist_ok=True)
                rec = distill.distill(student, teacher, train_loader, dev_loader,
                                      device, cfg, projection=None, out_dir=fin_dir)
                student.eval()
                report["final_recovery"] = {
                    "steps": final_steps,
                    "ce": rec.get("dev_loss"),
                    "kd": evaluate_kd(student, teacher, dev_loader, device, w,
                                      args.select_batches),
                }
                print(f"  final: KD {report['final_recovery']['kd']:.4f}  "
                      f"CE {report['final_recovery']['ce']:.4f}")
            break

        orig = alive[best]
        print(f"  removing layer {orig} (position {best}): KD {base_kd:.4f} -> "
              f"{best_kd:.4f} ({damage:+.4f})")
        remove_layer(student, best)
        alive.pop(best)
        removed.append(orig)

        is_last = rnd == args.n_layers - 1
        round_steps = final_steps if is_last else steps
        if is_last and final_steps != steps:
            print(f"  final cut -> extended recovery: {round_steps} steps")
        # hidden_weight stays 0: distill.py derives `valid` from
        # attention_mask[:, 0, 0, :], which is a prefix-query row and therefore
        # wrong once the block is applied. Depth pruning has no boundary
        # correspondence to match against anyway.
        cfg = distill.DistillConfig(
            steps=round_steps, lr=args.lr, grad_accum=args.grad_accum,
            kd_weight=args.kd_weight, ce_weight=args.ce_weight, hidden_weight=0.0,
            temperature=args.temperature, log_every=args.log_every,
            eval_every=args.eval_every, save_every=0, amp_dtype="bf16")
        rnd_dir = os.path.join(args.out_dir, f"round_{rnd+1:02d}_drop{orig}")
        os.makedirs(rnd_dir, exist_ok=True)
        rec = distill.distill(student, teacher, train_loader, dev_loader, device,
                              cfg, projection=None, out_dir=rnd_dir)
        student.eval()

        post_kd = evaluate_kd(student, teacher, dev_loader, device, w, args.select_batches)
        post_ce = rec.get("dev_loss", float("nan"))
        n_par = surgery.count_parameters(student)
        print(f"  after recovery: KD {post_kd:.4f} (was {best_kd:.4f} at surgery, "
              f"{base_kd:.4f} before)   CE {post_ce:.4f} vs teacher {teacher_ce:.4f}"
              f"   params {n_par/1e6:.1f}M")

        surgery.save_pruned(student, rnd_dir, tok)
        report["rounds"].append({
            "round": rnd + 1, "removed_original_index": orig,
            "layers_left": len(student.llm.layers), "params": n_par,
            "steps": round_steps,
            "kd_before": base_kd, "kd_after_surgery": best_kd,
            "kd_damage": damage, "kd_after_recovery": post_kd,
            "ce_after_recovery": post_ce,
            "ranking": [{"original": alive[i] if i < len(alive) else i,
                         "kd_damage": v - base_kd} for i, v in ranked],
            "removed_so_far": list(removed),
            "wall_seconds": time.time() - t0,
        })
        with open(os.path.join(args.out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)

    print(f"\nremoved {len(removed)} layers: {removed}")
    print(f"final: {len(student.llm.layers)} layers, "
          f"{surgery.count_parameters(student)/1e6:.1f}M params")
    surgery.save_pruned(student, args.out_dir, tok)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'report.json')}")
    return report


if __name__ == "__main__":
    main()
