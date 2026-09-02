#!/usr/bin/env python3
"""Where is this model prunable? Measured, before any repair or retraining.

Phase B asks which axis to cut first. The plan ranks them by projected parameter
savings, but the number that decides the order is *damage per unit of gain* --
and the plan's own §11 warns that post-surgery loss is only a proxy whose
correlation with post-recovery quality has to be verified rather than assumed.
This measures the proxy for every axis on the same data, so at least the ranking
is empirical.

Each intervention is applied by MASKING rather than by physically resizing:
zeroing an FFN neuron's `gate`/`up` rows makes its `down` contribution exactly
zero, zeroing an attention group's `o_proj` columns removes that head's
contribution exactly, and a dropped layer becomes an identity on the residual
stream. The loss is therefore identical to real surgery, without the bookkeeping.

Reported per intervention: weighted-CE delta on held-out data (no repair, no
retraining), parameters removed, and the share of LM compute removed.

    python scripts/p4_width_pruning/prune_survey.py --model models/p2
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # scripts/  (package root)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))      # repo root

from transformers import AutoTokenizer  # noqa: E402

from omnivoice.models.omnivoice import OmniVoice, _resolve_model_path  # noqa: E402
# These modules use package-relative imports, so they must be reached through
# the package rather than added to sys.path individually.
from p4_width_pruning.distill import evaluate_loss  # noqa: E402
from p4_width_pruning.manifest import (  # noqa: E402
    CodecManifestDataset, build_dataloader, build_processor, seed_everything,
)


def layers(model):
    return model.llm.layers


# ---------------------------------------------------------------------------
# Interventions (all reversible; each returns an undo callable)
# ---------------------------------------------------------------------------


def drop_layers(model, idxs):
    """Replace whole decoder layers with the identity on the residual stream."""
    saved = {}
    for i in idxs:
        lyr = layers(model)[i]
        saved[i] = lyr.forward

        # Qwen3DecoderLayer.forward returns a bare tensor in transformers 5.x
        # (older versions returned a tuple), and the next layer's norm consumes
        # it directly -- returning a tuple here fails one layer downstream.
        def passthrough(hidden_states, *a, **kw):
            return hidden_states

        lyr.forward = passthrough
    return lambda: [setattr(layers(model)[i], "forward", f) for i, f in saved.items()]


def mask_ffn(model, scores, keep_frac):
    """Zero the lowest-scoring FFN neurons in every layer."""
    saved = []
    for i, lyr in enumerate(layers(model)):
        n = lyr.mlp.gate_proj.weight.shape[0]
        k = int(round(n * keep_frac))
        drop = torch.argsort(scores[i])[: n - k]
        for mod, dim in ((lyr.mlp.gate_proj, 0), (lyr.mlp.up_proj, 0)):
            saved.append((mod.weight, mod.weight.data.clone()))
            mod.weight.data.index_fill_(dim, drop.to(mod.weight.device), 0.0)
    return lambda: [w.data.copy_(v) for w, v in saved]


def mask_groups(model, scores, keep_groups):
    """Zero whole GQA groups -- the unit of removal, since H % Kv must stay integral."""
    cfg = model.llm.config
    kv, h, hd = cfg.num_key_value_heads, cfg.num_attention_heads, cfg.head_dim
    per = h // kv
    saved = []
    for i, lyr in enumerate(layers(model)):
        drop = torch.argsort(scores[i])[: kv - keep_groups]
        o = lyr.self_attn.o_proj
        saved.append((o.weight, o.weight.data.clone()))
        for g in drop.tolist():
            for q in range(g * per, (g + 1) * per):
                o.weight.data[:, q * hd:(q + 1) * hd] = 0.0
    return lambda: [w.data.copy_(v) for w, v in saved]


# ---------------------------------------------------------------------------
# Importance scores (one calibration pass)
# ---------------------------------------------------------------------------


@torch.no_grad()
def importance(model, loader, device, max_batches):
    """FFN-neuron and attention-group importance.

    FFN neurons score `mean|h_i| * ||down_proj[:, i]||`: activation magnitude
    alone is not enough, because a large activation multiplied into a small
    output column moves the residual stream very little.

    Attention groups score `mean||head output|| * ||o_proj slice||`, the same
    contribution-norm idea applied to whole GQA groups.
    """
    L = len(layers(model))
    cfg = model.llm.config
    kv, h, hd = cfg.num_key_value_heads, cfg.num_attention_heads, cfg.head_dim
    per = h // kv
    ffn = [torch.zeros(l.mlp.gate_proj.weight.shape[0], device=device) for l in layers(model)]
    att = [torch.zeros(kv, device=device) for _ in range(L)]
    n = 0
    hooks = []

    def mlp_hook(i):
        def f(mod, inp, out):
            ffn[i] += inp[0].abs().mean(dim=tuple(range(inp[0].dim() - 1))).float()
        return f

    def attn_hook(i):
        def f(mod, inp, out):
            x = inp[0]                                  # [B, S, H*hd]
            v = x.reshape(*x.shape[:-1], h, hd).abs().mean(
                dim=tuple(range(x.dim() - 1))).mean(-1)  # per query head
            att[i] += v.view(kv, per).mean(-1).float()
        return f

    for i, lyr in enumerate(layers(model)):
        hooks.append(lyr.mlp.down_proj.register_forward_hook(mlp_hook(i)))
        hooks.append(lyr.self_attn.o_proj.register_forward_hook(attn_hook(i)))

    for b, batch in enumerate(loader):
        if b >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        model(input_ids=batch["input_ids"], audio_mask=batch["audio_mask"],
              attention_mask=batch["attention_mask"], position_ids=batch["position_ids"])
        n += 1
    for hk in hooks:
        hk.remove()

    for i, lyr in enumerate(layers(model)):
        ffn[i] = (ffn[i] / max(n, 1)) * lyr.mlp.down_proj.weight.norm(dim=0)
        o = lyr.self_attn.o_proj.weight
        gn = torch.stack([o[:, g * per * hd:(g + 1) * per * hd].norm()
                          for g in range(kv)]).to(device)
        att[i] = (att[i] / max(n, 1)) * gn
    return ffn, att


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--dev-manifest", default="data/dev_set.csv")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--eval-batches", type=int, default=6)
    ap.add_argument("--calib-batches", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--out", default="runs/prune_survey.json")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    seed_everything(42)
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    path = _resolve_model_path(args.model)
    model = OmniVoice.from_pretrained(path, train=True, dtype=dt,
                                      attn_implementation="sdpa").to(device).eval()
    tok = AutoTokenizer.from_pretrained(_resolve_model_path("k2-fsa/OmniVoice"))
    proc = build_processor(model.config, tok, deterministic=True)
    ds = CodecManifestDataset(args.dev_manifest, args.data_root, tokenizer=tok,
                              max_frames=args.max_frames)
    loader = build_dataloader(ds, proc, args.batch_tokens, shuffle=False, num_workers=0)

    cfg = model.llm.config
    L, d, I = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
    kv, h, hd = cfg.num_key_value_heads, cfg.num_attention_heads, cfg.head_dim
    attn_p = (d * h * hd) + 2 * (d * kv * hd) + (h * hd * d)
    mlp_p = 3 * d * I
    layer_p = attn_p + mlp_p
    total = sum(q.numel() for q in model.parameters())
    print(f"{args.model}: {total/1e6:.1f}M params, {L} layers, d={d}, I={I}, "
          f"{h}Q/{kv}KV heads\n  per layer {layer_p/1e6:.2f}M "
          f"(attn {attn_p/1e6:.2f}M + mlp {mlp_p/1e6:.2f}M), "
          f"transformer total {L*layer_p/1e6:.1f}M\n")

    base = evaluate_loss(model, loader, device, max_batches=args.eval_batches)
    print(f"baseline weighted CE: {base:.4f}\n")

    print("collecting importance scores ...", flush=True)
    ffn_s, att_s = importance(model, loader, device, args.calib_batches)

    rows = []

    def run(name, undo_fn, params, compute):
        loss = evaluate_loss(model, loader, device, max_batches=args.eval_batches)
        undo_fn()
        rows.append({"name": name, "loss": loss, "delta": loss - base,
                     "params_m": params / 1e6, "compute_frac": compute})
        print(f"  {name:28s} CE {loss:7.4f}  d{loss-base:+7.4f}   "
              f"-{params/1e6:6.1f}M  -{compute*100:5.1f}% compute", flush=True)

    print("--- DEPTH: drop one layer ---")
    for i in range(L):
        run(f"drop layer {i}", drop_layers(model, [i]), layer_p, 1 / L)

    print("\n--- DEPTH: drop contiguous spans ---")
    for lo, n in ((1, 4), (5, 4), (9, 4), (13, 4), (2, 8), (6, 8)):
        run(f"drop layers {lo}-{lo+n-1}", drop_layers(model, list(range(lo, lo + n))),
            n * layer_p, n / L)

    print("\n--- FFN: keep top fraction of neurons ---")
    for f in (0.75, 0.5, 0.375, 0.25):
        run(f"ffn keep {f:.3f} (I={int(I*f)})", mask_ffn(model, ffn_s, f),
            L * 3 * d * int(I * (1 - f)), (1 - f) * mlp_p / layer_p)

    print("\n--- ATTENTION: keep N of 8 GQA groups ---")
    for g in (6, 4, 2):
        run(f"groups {g}/8", mask_groups(model, att_s, g),
            L * attn_p * (kv - g) / kv, (kv - g) / kv * attn_p / layer_p)

    rows.sort(key=lambda r: r["delta"] / max(r["compute_frac"], 1e-9))
    print(f"\n{'='*76}\nBest damage-per-compute first  (CE delta per 1% of LM compute removed)"
          f"\n{'='*76}")
    print(f"{'intervention':30s} {'dCE':>8s} {'params':>9s} {'compute':>8s} {'dCE/1%':>9s}")
    for r in rows[:14]:
        print(f"{r['name']:30s} {r['delta']:+8.4f} {r['params_m']:8.1f}M "
              f"{r['compute_frac']*100:7.1f}% {r['delta']/max(r['compute_frac']*100,1e-9):9.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "baseline": base, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
