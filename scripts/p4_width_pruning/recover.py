#!/usr/bin/env python3
"""Recovery training only: no pruning, no layer selection.

`progressive_depth_prune.py` interleaves cut / recover / cut / recover. Sometimes
the cutting is already done and only the recovery is wanted -- the ladder ran with
a short per-round budget, the final model needs longer, and re-running the whole
ladder to get there would redo eight cuts to change one training run.

This is that last stage on its own. It takes a model that has ALREADY been pruned
(any layer count; the checkpoint's config carries it), trains it against a fixed
teacher under exactly the ladder's setup -- same prompt split, same optional
prefix block, same KD+CE mix, same CFG-teacher option -- and writes the result.

The teacher must be given explicitly. Defaulting it to `--model` would distil the
pruned model against itself, which trains towards an already-degraded target and
measures nothing; `progressive_depth_prune.py` can get away with that default only
because it loads the teacher before the first cut.

    python scripts/p4_width_pruning/recover.py \\
        --model runs/depth/round_04_drop10 --teacher models/p2 \\
        --out-dir runs/depth/extra --epochs 4 --kd-weight 0 --ce-weight 1.0 \\
        --lr 2.5e-5 --prefix-blocked --device cuda --batch-tokens 32768
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from torch.utils.data import Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from omnivoice.models.omnivoice import OmniVoice, _resolve_model_path  # noqa: E402
from p4_width_pruning import distill, surgery  # noqa: E402
from p4_width_pruning.manifest import (  # noqa: E402
    CodecManifestDataset, build_processor, seed_everything,
)
from p4_width_pruning.progressive_depth_prune import (  # noqa: E402
    DTYPES, FRAME_RATE, PromptSplitDataset, build_loader, evaluate_kd,
    make_cfg_teacher,
)


class MixedReferenceDataset(Dataset):
    """Half same-utterance prefix, half a separate reference clip by the same speaker.

    `PromptSplitDataset` always takes the prompt from the target utterance itself,
    which is the original training scheme but not the deployment one: real cloning
    supplies a different clip. Measured on the dev set, the pruned model's CE gap to
    p2 is ~11x larger under the cross-utterance layout (+0.124) than under the
    same-utterance one (+0.011), so the two are not interchangeable.

    Odd indices concatenate another utterance by the same speaker in front of the
    target -- codec and transcript both -- and mark the whole reference as the prompt,
    which reproduces the inference layout through the unmodified processor. Even
    indices keep the original scheme. Splitting by index rather than at random keeps
    `lengths` exact for the token-budget sampler.
    """

    def __init__(self, base, min_target_frames, max_prompt_ratio, ref_cap):
        self.base = base
        self.min_target_frames = min_target_frames
        self.max_prompt_ratio = max_prompt_ratio
        self.ref_cap = ref_cap
        self.by_speaker = defaultdict(list)
        for i, r in enumerate(base.rows):
            self.by_speaker[r["speaker_id"]].append(i)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        pool = [j for j in self.by_speaker[self.base.rows[i]["speaker_id"]] if j != i]
        if i % 2 and pool:
            ref = self.base[random.choice(pool)]
            clip = ref["audio_tokens"][:, : self.ref_cap]
            s["audio_tokens"] = torch.cat([clip, s["audio_tokens"]], dim=1)
            s["label"] = dict(s["label"],
                              text=f"{ref['label']['text']} {s['label']['text']}")
            s["_prompt_frames"] = clip.shape[1]
        else:
            T = s["audio_tokens"].shape[1]
            cap = max(0, T - self.min_target_frames)
            s["_prompt_frames"] = min(
                int(T * random.uniform(0.0, self.max_prompt_ratio)), cap)
        return s

    @property
    def lengths(self):
        # Odd rows may grow by up to ref_cap; budget for it even when the speaker
        # turns out to have no second utterance, so a batch can never overflow.
        return [n + (self.ref_cap if i % 2 else 0)
                for i, n in enumerate(self.base.lengths)]

    @property
    def hours(self):
        return self.base.hours


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("model / data")
    g.add_argument("--model", required=True,
                   help="the ALREADY PRUNED checkpoint to train further")
    g.add_argument("--teacher", required=True,
                   help="KD target and CE reference: the original, unpruned model")
    g.add_argument("--teacher-guidance-scale", type=float, default=0.0,
                   help="run the TEACHER with classifier-free guidance at this "
                        "scale (0 = plain single pass). Requires --temperature 1.")
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
    g.add_argument("--min-target-seconds", type=float, default=2.0)
    g.add_argument("--max-prompt-ratio", type=float, default=0.3)
    g.add_argument("--cross-utterance", action="store_true",
                   help="train on a 50/50 mix: half the samples keep the "
                        "same-utterance prefix, half use a separate clip by the "
                        "same speaker as the reference, which is the layout "
                        "inference uses. Applies to TRAINING only -- the dev set "
                        "stays same-utterance so its numbers remain comparable "
                        "with earlier runs.")
    g.add_argument("--ref-cap", type=int, default=150,
                   help="frames of the reference clip kept when --cross-utterance "
                        "is set (150 = 6s, matching stage 2)")
    g.add_argument("--prefix-blocked", action="store_true",
                   help="train and evaluate with prefix->target attention cut. "
                        "Must match how the model was pruned and how it is deployed.")

    g = p.add_argument_group("training")
    g.add_argument("--steps", type=int, default=0,
                   help="optimizer steps (0 = --epochs over the train set)")
    g.add_argument("--epochs", type=float, default=1.0)
    g.add_argument("--lr", type=float, default=2.5e-5)
    g.add_argument("--batch-tokens", type=int, default=16384)
    g.add_argument("--max-batch-size", type=int, default=128)
    g.add_argument("--grad-accum", type=int, default=1)
    g.add_argument("--kd-weight", type=float, default=1.0,
                   help="0 disables KD: training becomes pure CE and the teacher "
                        "is released after the reference eval.")
    g.add_argument("--kd-reverse", action="store_true",
                   help="minimise KL(student || teacher) instead of the forward "
                        "direction. Reported metrics stay FORWARD KL.")
    g.add_argument("--ce-weight", type=float, default=0.05)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--save-every", type=int, default=0)
    g.add_argument("--dev-batches", type=int, default=0)
    g.add_argument("--grad-checkpointing", action="store_true")
    args = p.parse_args(argv)

    if args.kd_weight == 0 and args.ce_weight == 0:
        raise SystemExit("both --kd-weight and --ce-weight are 0: no training signal")
    use_cfg = args.teacher_guidance_scale > 0
    if use_cfg and args.kd_weight == 0:
        raise SystemExit("--teacher-guidance-scale with --kd-weight 0 is pointless: "
                         "the CFG teacher costs two forwards per step and nothing "
                         "consumes its output.")
    if use_cfg and args.temperature != 1.0:
        raise SystemExit("--teacher-guidance-scale requires --temperature 1: the "
                         "mixed target is already a normalised distribution.")

    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)

    path = _resolve_model_path(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    student = OmniVoice.from_pretrained(
        path, train=True, dtype=DTYPES[args.dtype],
        attn_implementation=args.attn).to(device)
    if args.grad_checkpointing:
        student.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    teacher = OmniVoice.from_pretrained(
        _resolve_model_path(args.teacher), train=True,
        dtype=DTYPES[args.teacher_dtype],
        attn_implementation=args.attn).to(device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    w = torch.tensor(student.normalized_audio_codebook_weights, device=device)
    proc = build_processor(student.config, tok)
    ds_kw = {"tokenizer": tok, "min_frames": args.min_frames,
             "max_frames": args.max_frames}
    min_tgt = int(round(args.min_target_seconds * FRAME_RATE))
    wrap = lambda d: PromptSplitDataset(d, min_tgt, args.max_prompt_ratio)  # noqa: E731
    train_base = CodecManifestDataset(args.train_manifest, args.data_root, **ds_kw)
    train_ds = (MixedReferenceDataset(train_base, min_tgt, args.max_prompt_ratio,
                                      args.ref_cap)
                if args.cross_utterance else wrap(train_base))
    dev_ds = wrap(CodecManifestDataset(args.dev_manifest, args.data_root, **ds_kw))

    tfn = make_cfg_teacher(args.teacher_guidance_scale,
                           student.config.audio_mask_id) if use_cfg else None
    mk = lambda ds, sh, nw: build_loader(  # noqa: E731
        ds, proc, args.batch_tokens, args.max_batch_size, sh, args.seed, nw,
        args.prefix_blocked, cfg_teacher=use_cfg)
    train_loader = mk(train_ds, True, args.num_workers)
    # num_workers=0: the eval helpers pin the RNG, and worker processes draw their
    # masks outside that seeding.
    dev_loader = mk(dev_ds, False, 0)

    per_epoch = max(1, len(train_loader) // args.grad_accum)
    steps = args.steps or max(1, int(per_epoch * args.epochs))
    n_layers = len(student.llm.layers)
    print(f"device={device}  layers={n_layers}  "
          f"params={surgery.count_parameters(student)/1e6:.1f}M")
    print(f"train {len(train_ds)} utts ({train_ds.hours:.1f} h), "
          f"{per_epoch} steps/epoch -> {steps} steps ({steps/per_epoch:.2g} ep)")
    print(f"prompt split: <= {args.max_prompt_ratio:g} of the utterance, "
          f"target floor {args.min_target_seconds:g}s ({min_tgt} frames)")
    print(f"attention: {'PREFIX-BLOCKED' if args.prefix_blocked else 'full bidirectional'}")
    print(f"teacher: {args.teacher}"
          + (f", running CFG at w={args.teacher_guidance_scale:g}" if use_cfg else ""))
    print(f"loss: kd {args.kd_weight}{' (reverse)' if args.kd_reverse else ''} "
          f"+ ce {args.ce_weight}\n")

    teacher_ce = distill.evaluate_loss(teacher, dev_loader, device, args.dev_batches)
    start_ce = distill.evaluate_loss(student, dev_loader, device, args.dev_batches)
    start_kd = (evaluate_kd(student, teacher, dev_loader, device, w,
                            args.dev_batches, teacher_logits_fn=tfn)
                if args.kd_weight > 0 else None)
    print(f"teacher dev CE {teacher_ce:.4f}   student dev CE {start_ce:.4f} "
          f"({start_ce - teacher_ce:+.4f})"
          + (f"   KD {start_kd:.4f}" if start_kd is not None else ""))

    if args.kd_weight == 0:
        # Nothing after this reads it: the loss is CE-only and the reference eval
        # is done. Freeing it returns ~1.2 GiB (bf16) for batch size instead.
        del teacher
        teacher = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        print("teacher released after the reference eval (CE-only training)")
    print()

    t0 = time.time()
    cfg = distill.DistillConfig(
        steps=steps, lr=args.lr, grad_accum=args.grad_accum,
        kd_weight=args.kd_weight, kd_reverse=args.kd_reverse,
        ce_weight=args.ce_weight, hidden_weight=0.0,
        temperature=args.temperature, log_every=args.log_every,
        eval_every=args.eval_every, save_every=args.save_every, amp_dtype="bf16")
    rec = distill.distill(student, teacher, train_loader, dev_loader, device, cfg,
                          projection=None, out_dir=args.out_dir,
                          teacher_logits_fn=tfn)
    student.eval()

    end_ce = rec.get("dev_loss", float("nan"))
    end_kd = (evaluate_kd(student, teacher, dev_loader, device, w, args.dev_batches,
                          teacher_logits_fn=tfn) if args.kd_weight > 0 else None)
    print(f"\nCE {start_ce:.4f} -> {end_ce:.4f} ({end_ce - start_ce:+.4f})   "
          f"vs teacher {teacher_ce:.4f} ({end_ce - teacher_ce:+.4f})"
          + (f"   KD {start_kd:.4f} -> {end_kd:.4f}" if end_kd is not None else ""))

    surgery.save_pruned(student, args.out_dir, tok)
    report = {"args": vars(args), "layers": n_layers,
              "params": surgery.count_parameters(student), "steps": steps,
              "teacher_ce": teacher_ce, "ce_before": start_ce, "ce_after": end_ce,
              "kd_before": start_kd, "kd_after": end_kd,
              "wall_seconds": time.time() - t0}
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'report.json')}")
    return report


if __name__ == "__main__":
    main()
