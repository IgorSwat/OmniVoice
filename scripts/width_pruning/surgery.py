"""Weight surgery: RMSNorm gain folding, Q folding, truncation.

Implements sections 2, 3 and 4 of ``knowledge/width_pruning.md``.

The residual stream is a shared accumulating buffer that every module reads from
and writes into, so any orthogonal ``Q`` can be folded into the weights for free:
a read ``y = Wx`` becomes ``y = (WQ)x'`` and a write becomes ``(Q^T W)h``. The
skip connection survives because ``Q^T(a + b) = Q^T a + Q^T b``. Truncating to
the first ``k`` columns of ``Q`` is the only lossy step.

Preconditions this architecture satisfies natively: pre-norm, RMSNorm, and no
biases anywhere (``attention_bias: false``, no MLP bias, no norm bias). A bias on
the residual stream would break the algebra.
"""

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Module surgery helpers
# --------------------------------------------------------------------------


def _f64(t: torch.Tensor) -> torch.Tensor:
    """Detach to CPU float64.

    All the folding algebra runs in float64 for exactness, and it must run on
    CPU: MPS has no float64 support at all. The weight setters cast back to the
    module's own dtype and device, so this is the only place precision is
    chosen.
    """
    return t.detach().cpu().double()


def _set_linear(mod: nn.Linear, W: torch.Tensor) -> None:
    """Replace a Linear's weight, updating in/out_features to match."""
    assert mod.bias is None, "bias on the residual path breaks the rotation algebra"
    W = W.to(dtype=mod.weight.dtype, device=mod.weight.device).contiguous()
    mod.weight = nn.Parameter(W, requires_grad=mod.weight.requires_grad)
    mod.out_features, mod.in_features = W.shape


def _set_embedding(mod: nn.Embedding, W: torch.Tensor) -> None:
    W = W.to(dtype=mod.weight.dtype, device=mod.weight.device).contiguous()
    mod.weight = nn.Parameter(W, requires_grad=mod.weight.requires_grad)
    mod.num_embeddings, mod.embedding_dim = W.shape


def _set_norm(mod, k: int) -> None:
    """Reset an RMSNorm to unit gain at the new width.

    Only valid after :func:`fold_rmsnorm_gains` has moved the gain downstream.
    """
    w = torch.ones(k, dtype=mod.weight.dtype, device=mod.weight.device)
    mod.weight = nn.Parameter(w, requires_grad=mod.weight.requires_grad)


def _layers(model):
    return model.llm.layers


def _stream_readers(model):
    """Modules that READ the residual stream, i.e. take a d-wide input.

    Every one of these sits behind an RMSNorm, which is why they all take the
    dimension rescale in :func:`apply_rotation`.
    """
    mods = []
    for layer in _layers(model):
        mods += [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
        ]
    mods.append(model.audio_heads)
    return mods


def _stream_writers(model):
    """Modules that WRITE into the residual stream, i.e. produce a d-wide output."""
    mods = []
    for layer in _layers(model):
        mods += [layer.self_attn.o_proj, layer.mlp.down_proj]
    return mods


def _stream_embeddings(model):
    """Embeddings write into the stream; their weight is [num_embeddings, d]."""
    return [model.llm.embed_tokens, model.audio_embeddings]


# --------------------------------------------------------------------------
# Step 1: fold RMSNorm gains
# --------------------------------------------------------------------------


def fold_rmsnorm_gains(model) -> None:
    """Fold every RMSNorm gain into the linear layer that follows it.

    ``RMSNorm(x) = (x / rms(x)) * g``. The ``x / rms(x)`` part commutes with an
    orthogonal ``Q`` because ``||Q^T x|| = ||x||``, but the learnable gain is
    ``diag(g)``, which does not. Folding ``g`` downstream leaves pure RMS
    normalization, and only then does ``Q`` commute exactly.

    ``q_norm``/``k_norm`` are deliberately untouched: they normalize over
    ``head_dim`` (128, explicit in the config and decoupled from ``hidden_size``),
    not over the residual stream.
    """
    with torch.no_grad():
        for layer in _layers(model):
            g = _f64(layer.input_layernorm.weight)
            for m in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ):
                _set_linear(m, _f64(m.weight) * g.unsqueeze(0))
            layer.input_layernorm.weight.data.fill_(1.0)

            g = _f64(layer.post_attention_layernorm.weight)
            for m in (layer.mlp.gate_proj, layer.mlp.up_proj):
                _set_linear(m, _f64(m.weight) * g.unsqueeze(0))
            layer.post_attention_layernorm.weight.data.fill_(1.0)

        # The final norm's only consumer is audio_heads: this model has no
        # lm_head (the sole output projection is audio_heads [8200, d]).
        g = _f64(model.llm.norm.weight)
        _set_linear(
            model.audio_heads, _f64(model.audio_heads.weight) * g.unsqueeze(0)
        )
        model.llm.norm.weight.data.fill_(1.0)


def gains_are_folded(model, tol: float = 1e-5) -> bool:
    for layer in _layers(model):
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            if not torch.allclose(
                norm.weight.data, torch.ones_like(norm.weight.data), atol=tol
            ):
                return False
    return torch.allclose(
        model.llm.norm.weight.data, torch.ones_like(model.llm.norm.weight.data), atol=tol
    )


# --------------------------------------------------------------------------
# Steps 2 + 3: fold Q and truncate
# --------------------------------------------------------------------------


def apply_rotation(
    model,
    Q: torch.Tensor,
    k: Optional[int] = None,
    rms_rescale: bool = True,
) -> None:
    """Fold ``Q`` into every stream-facing weight and truncate to ``k`` columns.

    Read-side gets ``Q`` on the stream-facing side, write-side gets ``Q^T``. With
    PyTorch's ``[out, in]`` layout that means ``W <- W @ Qk`` for reads and
    ``W <- Qk^T @ W`` for writes; embeddings are writes whose weight is stored as
    ``[num_embeddings, d]``, so they take ``E @ Qk``.

    ``rms_rescale`` compensates the change in the RMSNorm normalizer. ``rms``
    averages over dimensions, so at the same vector norm the normalizer grows by
    ``sqrt(d/k)`` and the normalized output shrinks by ``sqrt(k/d)``. Read weights
    are therefore scaled by ``sqrt(d/k)``.

    (Note this differs from the direction stated in section 3 of the knowledge
    doc, which has the factor inverted. The exact factor is ``sqrt(d/k * r_l)``
    where ``r_l`` is the retention at that boundary; with ``r_l ~ 0.98`` the
    ``sqrt(r_l)`` term is a ~1% correction that the least-squares repair absorbs,
    so the constant is used here.)

    With ``k == d`` this is an exact no-op on the function computed — see
    :func:`verify_roundtrip`.
    """
    d = model.config.llm_config.hidden_size
    assert Q.shape == (d, d), f"Q must be [{d}, {d}], got {tuple(Q.shape)}"
    k = k or d
    assert 0 < k <= d

    Qk = _f64(Q[:, :k])
    scale = math.sqrt(d / k) if rms_rescale else 1.0

    with torch.no_grad():
        for mod in _stream_embeddings(model):
            _set_embedding(mod, _f64(mod.weight) @ Qk)

        for mod in _stream_readers(model):
            _set_linear(mod, (_f64(mod.weight) @ Qk) * scale)

        for mod in _stream_writers(model):
            _set_linear(mod, Qk.T @ _f64(mod.weight))

        for layer in _layers(model):
            _set_norm(layer.input_layernorm, k)
            _set_norm(layer.post_attention_layernorm, k)
        _set_norm(model.llm.norm, k)

    _set_hidden_size(model, k)


def _set_hidden_size(model, k: int) -> None:
    model.config.llm_config.hidden_size = k
    model.llm.config.hidden_size = k
    for layer in _layers(model):
        if hasattr(layer.self_attn, "config"):
            layer.self_attn.config.hidden_size = k
        if hasattr(layer.mlp, "config"):
            layer.mlp.config.hidden_size = k


def prune_width(
    model,
    Q: torch.Tensor,
    k: int,
    rms_rescale: bool = True,
    fold_gains: bool = True,
) -> None:
    """Full surgery: fold gains, fold Q, truncate. Mutates ``model`` in place."""
    if fold_gains and not gains_are_folded(model):
        fold_rmsnorm_gains(model)
    apply_rotation(model, Q, k=k, rms_rescale=rms_rescale)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


@torch.no_grad()
def verify_roundtrip(
    model,
    batch,
    seed: int = 0,
    tol: float = 2e-2,
) -> Tuple[bool, float]:
    """Assert a full-rank Q round-trip is a no-op BEFORE anything is truncated.

    Runs the model, applies gain folding plus a full-rank random orthogonal ``Q``
    to a deep copy, and compares logits. This is the unit test that catches an
    inverted transpose or a missed module; if it fails, nothing downstream is
    trustworthy.

    Returns ``(passed, max_relative_error)``. The default tolerance is loose
    because the comparison runs at the model's own dtype — run it in float32 for
    a tight check.
    """
    from .calibration import _audio_logits, _hidden_states

    model.eval()
    _, ref_last = _hidden_states(model, batch, output_hidden_states=False)
    ref = _audio_logits(model, ref_last).float()

    probe = copy.deepcopy(model)
    d = probe.config.llm_config.hidden_size
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=gen, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)

    fold_rmsnorm_gains(probe)
    apply_rotation(probe, Q, k=d, rms_rescale=True)
    probe.eval()
    _, got_last = _hidden_states(probe, batch, output_hidden_states=False)
    got = _audio_logits(probe, got_last).float()

    denom = ref.abs().max().clamp(min=1e-6)
    err = float((got - ref).abs().max() / denom)
    del probe
    return err <= tol, err


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_pruned(model, out_dir: str, tokenizer=None) -> None:
    """Write the student as a stock checkpoint.

    With a single global ``Q`` the student is an ordinary ``Qwen3Model`` at the
    new ``hidden_size``, loadable by HuggingFace with no custom code. ``head_dim``
    stays 128: it is explicit in ``Qwen3Config`` and decoupled from
    ``hidden_size``, so RoPE and ``q_norm``/``k_norm`` are untouched and the
    attention projections simply become wider than the residual stream.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# Block grouping (section 7): per-block Q with adapters at the junctions
# --------------------------------------------------------------------------


def block_layer_ranges(splits, num_layers: int):
    """``[21]`` -> ``[(0, 20), (21, 27)]``. Splits are the first layer of a block."""
    edges = [0] + sorted(int(s) for s in splits) + [num_layers]
    return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]


def block_boundaries(lo: int, hi: int, num_boundaries: int):
    """Residual boundaries a layer block touches: it reads ``lo..hi`` and writes ``hi+1``.

    Blocks overlap by one boundary at each junction on purpose — the shared
    boundary is where the adapter converts between bases, so both neighbouring
    bases should represent it well.
    """
    return list(range(lo, min(hi + 2, num_boundaries)))


def _fold_block(model, layers, Qk, scale):
    with torch.no_grad():
        for layer in layers:
            for m in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                      layer.self_attn.v_proj, layer.mlp.gate_proj, layer.mlp.up_proj):
                _set_linear(m, (_f64(m.weight) @ Qk) * scale)
            for m in (layer.self_attn.o_proj, layer.mlp.down_proj):
                _set_linear(m, Qk.T @ _f64(m.weight))
            _set_norm(layer.input_layernorm, Qk.shape[1])
            _set_norm(layer.post_attention_layernorm, Qk.shape[1])


class _AdapterPreHook:
    """Applies a residual-stream basis change to a decoder layer's input.

    A pre-hook rather than a wrapper module: `Qwen3Model.forward` reaches into
    its layers for attributes like `attention_type`, and wrapping would hide
    them. The adapter itself is registered on the model as `width_adapters` so
    it is a real parameter — trainable during recovery and saved in the
    state dict.
    """

    def __init__(self, adapter):
        self.adapter = adapter

    def __call__(self, module, args, kwargs):
        if args:
            return (self.adapter(args[0]),) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = self.adapter(kwargs["hidden_states"])
        return args, kwargs


def install_adapters(model, adapters):
    """``adapters``: ``{first_layer_of_block: [k, k] matrix}``, applied to its input."""
    ref = model.llm.layers[0].self_attn.q_proj.weight
    mods = nn.ModuleList()
    for _, A in sorted(adapters.items()):
        lin = nn.Linear(A.shape[1], A.shape[0], bias=False)
        with torch.no_grad():
            lin.weight.copy_(A.to(dtype=ref.dtype))
        mods.append(lin.to(device=ref.device, dtype=ref.dtype))
    model.width_adapters = mods
    handles = []
    for i, li in enumerate(sorted(adapters)):
        handles.append(
            model.llm.layers[li].register_forward_pre_hook(
                _AdapterPreHook(model.width_adapters[i]), with_kwargs=True
            )
        )
    model._adapter_layers = sorted(adapters)
    model._adapter_handles = handles
    return model.width_adapters


def reinstall_adapters(model):
    """Re-attach hooks after loading a checkpoint that already carries adapters."""
    if not hasattr(model, "width_adapters"):
        return
    for i, li in enumerate(model._adapter_layers):
        model.llm.layers[li].register_forward_pre_hook(
            _AdapterPreHook(model.width_adapters[i]), with_kwargs=True
        )


def prune_width_blocks(model, Qs, ranges, k, rms_rescale=True, fold_gains=True):
    """Fold one basis per contiguous layer block and truncate to ``k``.

    ``Qs[b]`` is the full ``[d, d]`` rotation for the layers in ``ranges[b]``.
    Embeddings write the stream in block 0's basis; ``audio_heads`` reads it in
    the last block's. At each junction an adapter converts the residual stream
    from the previous basis to the next: ``A = Q_next[:, :k]^T @ Q_prev[:, :k]``.

    With one block this is exactly :func:`apply_rotation`.
    """
    if fold_gains and not gains_are_folded(model):
        fold_rmsnorm_gains(model)
    d = model.config.llm_config.hidden_size
    scale = math.sqrt(d / k) if rms_rescale else 1.0
    Qks = [_f64(Q[:, :k]) for Q in Qs]

    with torch.no_grad():
        for mod in _stream_embeddings(model):          # written in block 0's basis
            _set_embedding(mod, _f64(mod.weight) @ Qks[0])
        for b, (lo, hi) in enumerate(ranges):
            _fold_block(model, model.llm.layers[lo:hi + 1], Qks[b], scale)
        _set_linear(model.audio_heads,                 # reads the last block's basis
                    (_f64(model.audio_heads.weight) @ Qks[-1]) * scale)
        _set_norm(model.llm.norm, k)

    adapters = {ranges[b][0]: (Qks[b].T @ Qks[b - 1]) for b in range(1, len(ranges))}
    _set_hidden_size(model, k)
    if adapters:
        install_adapters(model, adapters)
    return adapters
