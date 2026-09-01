#!/usr/bin/env python3
"""Pick `--ce-weight` by measuring gradients, not by guessing from loss values.

Loss magnitudes are the wrong basis for balancing two terms: what actually
competes during training is their *gradients*. This probe backprops each term
separately from the same forward pass and reports, per checkpoint:

  ||g_kd||, ||g_ce||   the pull each term exerts on the weights
  cos(g_kd, g_ce)      whether CE is redundant with KD (~1) or adds a genuinely
                       different direction (~0), which is the entire reason to add it
  suggested lambda     lambda = share * ||g_kd|| / ||g_ce||, so that
                       ||lambda * g_ce|| / ||g_kd|| == share

Run it at BOTH ends of training -- the teacher (student init, where KD is
largest) and a finished student (where KD has shrunk). CE is bounded below by
the task's intrinsic entropy and barely decays, while KD decays a lot, so a
weight balanced at init drifts toward CE dominance by the end. The two numbers
bracket that drift.

    python scripts/prefix_blocking/probe_loss_balance.py \
        --student k2-fsa/OmniVoice --student runs/prefix_blocked
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from omnivoice.models.omnivoice import OmniVoice, _resolve_model_path  # noqa: E402
from train_prefix_blocked import (  # noqa: E402
    DTYPES, ClonePairDataset, CloneCollator, build_masks, logits_of,
    weighted_ce, weighted_kd, to_device,
)


def grad_stats(student, teacher, batch, w, temperature):
    """Separate gradients for each loss term from one shared forward pass."""
    full = build_masks(batch["valid"], batch["prefix_len"], False)
    blocked = build_masks(batch["valid"], batch["prefix_len"], True)
    with torch.no_grad():
        t_logits = logits_of(teacher, batch, full)
    s_logits = logits_of(student, batch, blocked)

    kd = weighted_kd(s_logits, t_logits, batch["labels"], w, temperature)
    ce = weighted_ce(s_logits, batch["labels"], w)

    params = [p for p in student.parameters() if p.requires_grad]
    g_kd = torch.autograd.grad(kd, params, retain_graph=True, allow_unused=True)
    g_ce = torch.autograd.grad(ce, params, allow_unused=True)

    sq_kd = sq_ce = dot = 0.0
    for a, b in zip(g_kd, g_ce):
        if a is None or b is None:
            continue
        a32, b32 = a.float(), b.float()
        sq_kd += float(a32.pow(2).sum())
        sq_ce += float(b32.pow(2).sum())
        dot += float((a32 * b32).sum())
    return {"kd": float(kd.detach()), "ce": float(ce.detach()),
            "n_kd": sq_kd ** 0.5, "n_ce": sq_ce ** 0.5, "dot": dot}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default="k2-fsa/OmniVoice")
    p.add_argument("--student", action="append", default=None,
                   help="repeatable; default probes the teacher-init and the "
                        "trained prefix-blocked checkpoint")
    p.add_argument("--dev-manifest", default="data/dev_set.csv")
    p.add_argument("--dev-ref-manifest", default="data/dataset_without_dev.csv")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    p.add_argument("--ref-cap", type=int, default=150)
    p.add_argument("--max-frames", type=int, default=750)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--shares", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    students = args.student or ["k2-fsa/OmniVoice", "runs/prefix_blocked"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    dt = DTYPES[args.dtype]

    tpath = _resolve_model_path(args.teacher)
    tok = AutoTokenizer.from_pretrained(tpath)
    teacher = OmniVoice.from_pretrained(tpath, train=True, dtype=dt,
                                        attn_implementation="sdpa").to(device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    coll = CloneCollator(tok, teacher.config.num_audio_codebook,
                         teacher.config.audio_mask_id)
    ds = ClonePairDataset(args.dev_manifest, args.data_root,
                          ref_manifest=args.dev_ref_manifest,
                          max_frames=args.max_frames, ref_cap=args.ref_cap)
    w = torch.tensor(teacher.normalized_audio_codebook_weights, device=device)
    print(f"device={device} dtype={args.dtype}  dev pairs {len(ds)}  "
          f"{args.batches} batches x {args.batch_size}\n")

    results = {}
    for spec in students:
        student = OmniVoice.from_pretrained(_resolve_model_path(spec), train=True,
                                            dtype=dt,
                                            attn_implementation="sdpa").to(device)
        student.eval()  # no dropout in this model, but be explicit
        # Identical batches for every checkpoint.
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=coll, num_workers=0)

        acc = {"kd": 0.0, "ce": 0.0, "n_kd": 0.0, "n_ce": 0.0, "cos": 0.0}
        n = 0
        for i, batch in enumerate(dl):
            if i >= args.batches:
                break
            st = grad_stats(student, teacher, to_device(batch, device), w,
                            args.temperature)
            acc["kd"] += st["kd"]; acc["ce"] += st["ce"]
            acc["n_kd"] += st["n_kd"]; acc["n_ce"] += st["n_ce"]
            acc["cos"] += st["dot"] / max(st["n_kd"] * st["n_ce"], 1e-12)
            n += 1
            print(f"  {spec}: batch {i+1}/{args.batches}", end="\r", flush=True)
        results[spec] = {k: v / n for k, v in acc.items()}
        del student
        if device.type == "mps":
            torch.mps.empty_cache()

    print(" " * 50, end="\r")
    print(f"{'checkpoint':28s} {'KD':>8s} {'CE':>8s} {'|g_kd|':>10s} "
          f"{'|g_ce|':>10s} {'ratio':>8s} {'cos':>7s}")
    print("-" * 84)
    for spec, r in results.items():
        print(f"{spec:28s} {r['kd']:8.4f} {r['ce']:8.4f} {r['n_kd']:10.4f} "
              f"{r['n_ce']:10.4f} {r['n_ce']/r['n_kd']:8.2f} {r['cos']:7.3f}")

    print(f"\nlambda for a target gradient share  (lambda = share * |g_kd| / |g_ce|)")
    print(f"{'share of |g_kd|':20s} " + " ".join(f"{s:>28s}" for s in results))
    for share in args.shares:
        row = " ".join(f"{share * r['n_kd'] / r['n_ce']:28.4f}"
                       for r in results.values())
        print(f"{share*100:17.0f}%  {row}")


if __name__ == "__main__":
    main()
