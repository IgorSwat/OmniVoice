#!/usr/bin/env python3
"""Progressive width pruning of OmniVoice with least-squares repair and KD recovery.

Shrinks the Qwen3 backbone's ``hidden_size`` one rung at a time. Each rung:

  1. Re-calibrate the residual stream on the CURRENT student (importance is only
     locally valid — a Q fit to the 1024-wide model does not describe the stream
     of an 896-wide one).
  2. Build the rotation Q from the trace-normalized, audio-position-only second
     moments, and truncate to k.
  3. Closed-form least-squares repair of the write-side projections against the
     pre-cut model.
  4. KD recovery against the ORIGINAL teacher.
  5. Gate on dev loss; stop the ladder if the rung does not recover.

Everything is a stock ``Qwen3Model`` at the new ``hidden_size`` — one global Q,
no adapters, no architecture change. Measured on this checkpoint, one global Q at
k=832 (0.9649 worst-case retention) matches three blocks at k=704 (0.9639) while
keeping the student loadable by HuggingFace, so width is a cheaper currency than
adapters.

Example
-------
    python scripts/progressive_width_prune.py \
        --widths 896 800 704 \
        --train-manifest data/dataset_without_dev.csv \
        --dev-manifest data/dev_set.csv \
        --out-dir runs/ladder_v1 \
        --steps 4000 --batch-tokens 16384 --device cuda

See ``knowledge/width_pruning.md`` for the theory and the measured numbers.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer  # noqa: E402

from omnivoice.models.omnivoice import OmniVoice, _resolve_model_path  # noqa: E402
from width_pruning import calibration, distill, repair, surgery  # noqa: E402
from width_pruning.manifest import (  # noqa: E402
    CodecManifestDataset,
    build_dataloader,
    build_processor,
    seed_everything,
    to_device,
)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = p.add_argument_group("model / data")
    g.add_argument("--model", default="k2-fsa/OmniVoice")
    g.add_argument("--train-manifest", default="data/dataset_without_dev.csv")
    g.add_argument("--dev-manifest", default="data/dev_set.csv")
    g.add_argument("--data-root", default="data")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--device", default=None, help="cuda / mps / cpu (auto by default)")
    g.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    g.add_argument("--attn", default="sdpa")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument(
        "--min-frames", type=int, default=50, help="drop utterances shorter than this"
    )
    g.add_argument(
        "--max-frames",
        type=int,
        default=2000,
        help="drop utterances longer than this. The corpus already tops out at "
        "750 frames (30 s), so the default never binds; lower it only to trade "
        "data for memory, since activation cost per token grows with sequence "
        "length.",
    )
    g.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[896, 800, 704],
        help="target hidden_size per rung, descending",
    )
    g.add_argument(
        "--gate-rel",
        type=float,
        default=0.01,
        help="stop the ladder if a rung's dev loss exceeds the teacher's by more "
        "than this relative margin after recovery",
    )
    g.add_argument(
        "--continue-on-gate-fail",
        action="store_true",
        help="record the gate failure but keep descending anyway",
    )

    g = p.add_argument_group("calibration / basis")
    g.add_argument("--calib-samples", type=int, default=192)
    g.add_argument("--calib-batch-tokens", type=int, default=4096)
    g.add_argument(
        "--basis",
        default="pca",
        choices=["pca", "fisher", "selection"],
        help="pca: uncentered PCA of the audio-position second moment. "
        "fisher: output-sensitivity weighted (one backward pass, ~4x forward). "
        "selection: axis-aligned channel selection, no rotation at all.",
    )
    g.add_argument(
        "--no-trace-normalize",
        action="store_true",
        help="reproduce the naive raw-sum failure (0.935 -> 0.719 retention)",
    )
    g.add_argument("--no-rms-rescale", action="store_true")
    g.add_argument("--skip-roundtrip-check", action="store_true")
    g.add_argument(
        "--splits",
        type=int,
        nargs="*",
        default=[],
        help="layer indices that start a new block (section 7). Each block gets "
        "its own Q and a [k, k] adapter reconciles the residual stream at each "
        "junction. '--splits 21' costs 0.5M params (0.12%% of a 704-wide "
        "student) and ~0.3%% of forward compute, and lifts worst-boundary "
        "retention at k=704 from 0.9353 to 0.9542; '--splits 21 25' reaches "
        "0.9642. Empty (default) = one global Q, and the student stays a stock "
        "Qwen3Model.",
    )

    g = p.add_argument_group("repair")
    g.add_argument("--repair-mode", default="sequential", choices=["sequential", "joint", "none"])
    g.add_argument("--repair-positions", default="all", choices=["all", "audio"])
    g.add_argument("--repair-ridge", type=float, default=1e-4)

    g = p.add_argument_group("recovery")
    g.add_argument("--steps", type=int, default=4000)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--batch-tokens", type=int, default=16384)
    g.add_argument("--max-batch-size", type=int, default=64)
    g.add_argument("--grad-accum", type=int, default=1)
    g.add_argument("--kd-weight", type=float, default=1.0)
    g.add_argument("--ce-weight", type=float, default=0.0)
    g.add_argument(
        "--hidden-weight",
        type=float,
        default=0.0,
        help="per-boundary hidden-state matching against the rotated teacher "
        "stream; exact at initialization, drifts as training proceeds",
    )
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--save-every", type=int, default=0)
    g.add_argument(
        "--skip-recovery",
        action="store_true",
        help="surgery + repair only; useful for sweeping k on the cheap proxy",
    )
    g.add_argument("--max-dev-batches", type=int, default=0)
    g.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="recompute layer activations in the backward pass: ~5x less "
        "activation memory for ~30%% more compute. The cheapest way to raise "
        "--batch-tokens.",
    )
    g.add_argument(
        "--teacher-dtype",
        default=None,
        choices=list(DTYPES),
        help="teacher precision (default: same as --dtype). It is frozen, so "
        "bf16 costs nothing and saves ~1.2 GB.",
    )

    return p.parse_args(argv)


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path, dtype, attn, device):
    model = OmniVoice.from_pretrained(
        path, train=True, dtype=dtype, attn_implementation=attn
    )
    return model.to(device).eval()


def report_spectrum(moments, k, tag=""):
    """Print the diagnostics that decide whether k is above the intrinsic dimension.

    Retention is a CAPACITY signal, not a damage estimate: under prune-then-retrain
    it bounds initialization damage only. Do not gate on a 99% threshold.
    """
    d = calibration.diagnostics(moments, k)
    own = d["retention_own_basis"]
    nb = len(own)
    deep = slice(max(nb - 10, 0), nb - 1)
    print(f"  [{tag}] per-boundary retention @k={k} (own basis):")
    print(f"    boundaries 0-{deep.start - 1}: {own[: deep.start].min():.4f} - {own[: deep.start].max():.4f}")
    print(f"    boundaries {deep.start}-{nb - 2}: {own[deep].min():.4f} - {own[deep].max():.4f}  <- worst band")
    print(f"    boundary {nb - 1} (post final norm): {own[-1]:.4f}")
    print(
        f"    mean-share deep {d['mean_share'][deep].min():.2f}-{d['mean_share'][deep].max():.2f}"
        f"   participation-ratio deep {d['participation_ratio'][deep].min():.1f}"
        f"-{d['participation_ratio'][deep].max():.1f} (of {moments.d})"
    )
    return d


def build_q(moments, args, k):
    if args.basis == "selection":
        Q = calibration.channel_selection(
            moments, k, trace_normalize=not args.no_trace_normalize
        )
    else:
        Q = calibration.compute_rotation(
            moments, trace_normalize=not args.no_trace_normalize
        )
    return Q


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = DTYPES[args.dtype]

    print(f"device={device} dtype={args.dtype} attn={args.attn}")
    path = _resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)

    teacher_dtype = DTYPES[args.teacher_dtype] if args.teacher_dtype else dtype
    teacher = load_model(path, teacher_dtype, args.attn, device)
    student = load_model(path, dtype, args.attn, device)
    if args.grad_checkpointing:
        student.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("gradient checkpointing: enabled on the student backbone")
    d0 = teacher.config.llm_config.hidden_size
    print(
        f"teacher: hidden_size={d0} layers={teacher.config.llm_config.num_hidden_layers} "
        f"params={surgery.count_parameters(teacher) / 1e6:.1f}M"
    )

    # Gain folding is exactly function-preserving, so folding the KD teacher costs
    # nothing and keeps its residual basis comparable to the student's for the
    # optional hidden-state matching term.
    surgery.fold_rmsnorm_gains(teacher)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    processor = build_processor(student.config, tokenizer)

    ds_kw = {
        "tokenizer": tokenizer,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
    }
    calib_ds = CodecManifestDataset(
        args.dev_manifest, args.data_root, limit=args.calib_samples, **ds_kw
    )
    dev_ds = CodecManifestDataset(args.dev_manifest, args.data_root, **ds_kw)
    # Building the training set scans every codec header to fill the length
    # cache (~300k files on the first run), so skip it entirely when there is no
    # recovery to run -- that is what keeps the --skip-recovery sweep cheap.
    train_ds = (
        None
        if args.skip_recovery
        else CodecManifestDataset(args.train_manifest, args.data_root, **ds_kw)
    )
    train_desc = (
        "skipped"
        if train_ds is None
        else f"{len(train_ds)} utts ({train_ds.hours:.1f} h, {train_ds.dropped} dropped)"
    )
    print(f"data: train {train_desc}  dev {len(dev_ds)}  calib {len(calib_ds)}")
    if train_ds is not None:
        print(
            f"      longest padded sequence in train: {max(train_ds.lengths)} tokens"
        )

    def loader(ds, batch_tokens, shuffle, workers=None):
        return build_dataloader(
            ds,
            processor,
            batch_tokens=batch_tokens,
            max_batch_size=args.max_batch_size,
            shuffle=shuffle,
            seed=args.seed,
            num_workers=args.num_workers if workers is None else workers,
        )

    calib_loader = loader(calib_ds, args.calib_batch_tokens, False)
    # num_workers=0: evaluate_loss pins the RNG so every eval sees identical
    # masks, and worker processes draw outside that seeding.
    dev_loader = loader(dev_ds, args.calib_batch_tokens, False, workers=0)
    train_loader = (
        None if train_ds is None else loader(train_ds, args.batch_tokens, True)
    )

    if not args.skip_roundtrip_check:
        print("verifying a full-rank Q round-trip is a no-op...")
        batch = to_device(next(iter(calib_loader)), device)
        ok, err = surgery.verify_roundtrip(student, batch)
        print(f"  max relative logit error: {err:.3e}  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(
                "round-trip check failed: the rotation is not being folded "
                "consistently. Nothing downstream is trustworthy; fix this first."
            )

    teacher_dev = distill.evaluate_loss(
        teacher, dev_loader, device, args.max_dev_batches
    )
    print(f"teacher dev loss: {teacher_dev:.4f}")

    report = {
        "args": vars(args),
        "teacher": {
            "hidden_size": d0,
            "params": surgery.count_parameters(teacher),
            "dev_loss": teacher_dev,
        },
        "rungs": [],
    }
    projection = None

    for rung, k in enumerate(args.widths):
        d = student.config.llm_config.hidden_size
        if k >= d:
            print(f"skipping rung k={k}: not smaller than current width {d}")
            continue
        print(f"\n=== rung {rung + 1}/{len(args.widths)}: {d} -> {k} ===")
        rung_dir = os.path.join(args.out_dir, f"rung_{k}")
        os.makedirs(rung_dir, exist_ok=True)
        t_rung = time.time()

        print(f"[1/5] calibrating on the current {d}-wide student "
              f"({args.basis}{' , Fisher-weighted' if args.basis == 'fisher' else ''})")
        moments = calibration.calibrate(
            student,
            calib_loader,
            device,
            fisher=(args.basis == "fisher"),
        )
        moments.save(os.path.join(rung_dir, "moments.pt"))
        diag = report_spectrum(moments, k, tag=f"d={d}")

        n_layers = student.config.llm_config.num_hidden_layers
        ranges = surgery.block_layer_ranges(args.splits, n_layers)
        print(f"[2/5] building Q and truncating ({len(ranges)} block(s), "
              f"{len(ranges) - 1} adapter(s))")
        Qs, rets = [], []
        for lo, hi in ranges:
            bnds = surgery.block_boundaries(lo, hi, moments.num_boundaries)
            Qb = build_q(moments, args, k) if len(ranges) == 1 else \
                calibration.compute_rotation(
                    moments, trace_normalize=not args.no_trace_normalize,
                    boundaries=bnds)
            Qs.append(Qb)
            rets.append(calibration.retention(moments, Qb, k)[bnds])
            if len(ranges) > 1:
                print(f"    block layers {lo}-{hi} (boundaries {bnds[0]}-{bnds[-1]}): "
                      f"min retention {rets[-1].min():.4f}")
        ret = np.concatenate(rets) if len(ranges) > 1 else calibration.retention(moments, Qs[0], k)
        print(
            f"  retention @k={k}: min {ret.min():.4f}  median {np.median(ret):.4f}"
        )

        pre_cut = copy.deepcopy(student)
        pre_cut.eval()
        adapters = surgery.prune_width_blocks(
            student, Qs, ranges, k, rms_rescale=not args.no_rms_rescale
        )
        if adapters:
            print(f"  installed {len(adapters)} adapter(s) at layers "
                  f"{sorted(adapters)} ({k * k / 1e6:.2f}M params each)")
        n_params = surgery.count_parameters(student)
        print(
            f"  student now hidden_size={k}  params={n_params / 1e6:.1f}M "
            f"({100 * (1 - n_params / report['teacher']['params']):.1f}% smaller than teacher)"
        )

        repair_report = {}
        if args.repair_mode != "none":
            print(f"[3/5] least-squares repair ({args.repair_mode})")
            repair_report = repair.repair(
                pre_cut,
                student,
                calib_loader,
                device,
                Qs if len(Qs) > 1 else Qs[0],
                k,
                ranges=ranges,
                mode=args.repair_mode,
                positions=args.repair_positions,
                ridge_rel=args.repair_ridge,
            )
        del pre_cut
        if device.type == "cuda":
            torch.cuda.empty_cache()

        post_surgery_dev = distill.evaluate_loss(
            student, dev_loader, device, args.max_dev_batches
        )
        print(
            f"[4/5] post-surgery dev loss: {post_surgery_dev:.4f} "
            f"(teacher {teacher_dev:.4f}, "
            f"{100 * (post_surgery_dev - teacher_dev) / teacher_dev:+.1f}%)"
        )

        # Hidden-state matching maps the teacher's stream into the student's.
        # With blocks each boundary has its own basis, so only the last block's
        # is a single global map; fall back to it and note the approximation.
        Qref = Qs[-1]
        projection = Qref[:, :k] if projection is None else projection @ Qref[:, :k]

        recovery = {}
        if not args.skip_recovery:
            print(f"[5/5] KD recovery against the original teacher ({args.steps} steps)")
            cfg = distill.DistillConfig(
                steps=args.steps,
                lr=args.lr,
                grad_accum=args.grad_accum,
                kd_weight=args.kd_weight,
                ce_weight=args.ce_weight,
                hidden_weight=args.hidden_weight,
                temperature=args.temperature,
                eval_every=args.eval_every,
                log_every=args.log_every,
                save_every=args.save_every,
                amp_dtype="bf16",
            )
            recovery = distill.distill(
                student,
                teacher,
                train_loader,
                dev_loader,
                device,
                cfg,
                projection=projection,
                out_dir=rung_dir,
            )
            student.eval()
        else:
            print("[5/5] recovery skipped (--skip-recovery)")

        final_dev = recovery.get("dev_loss", post_surgery_dev)
        gap = (final_dev - teacher_dev) / teacher_dev
        passed = gap <= args.gate_rel
        print(
            f"  rung {k}: dev loss {final_dev:.4f} vs teacher {teacher_dev:.4f} "
            f"({gap:+.2%})  gate {'PASS' if passed else 'FAIL'} "
            f"(threshold {args.gate_rel:+.2%})"
        )

        surgery.save_pruned(student, rung_dir, tokenizer)
        torch.save(
            {"Q": Qs[0] if len(Qs) == 1 else Qs, "k": k, "ranges": ranges,
             "splits": list(args.splits), "projection": projection,
             "adapters": {i: a for i, a in adapters.items()} if adapters else {}},
            os.path.join(rung_dir, "rotation.pt"),
        )

        report["rungs"].append(
            {
                "k": k,
                "from": d,
                "params": n_params,
                "splits": list(args.splits),
                "num_adapters": len(ranges) - 1,
                "retention_min": float(ret.min()),
                "retention_median": float(np.median(ret)),
                "retention_per_boundary": ret.tolist(),
                "diagnostics": {kk: vv.tolist() for kk, vv in diag.items()},
                "repair": repair_report,
                "post_surgery_dev_loss": post_surgery_dev,
                "recovered_dev_loss": final_dev,
                "gap_vs_teacher": gap,
                "gate_passed": bool(passed),
                "recovery": {
                    kk: vv for kk, vv in recovery.items() if kk != "history"
                },
                "recovery_history": recovery.get("history", []),
                "wall_seconds": time.time() - t_rung,
            }
        )
        with open(os.path.join(args.out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)

        if not passed and not args.continue_on_gate_fail:
            print(
                f"\nstopping: rung {k} did not recover to within {args.gate_rel:.1%} "
                f"of the teacher. Re-run with a longer --steps, a larger k, or "
                f"--continue-on-gate-fail to descend anyway."
            )
            break

    print(f"\nwrote {os.path.join(args.out_dir, 'report.json')}")
    return report


if __name__ == "__main__":
    main()
