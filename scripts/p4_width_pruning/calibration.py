"""Residual-stream calibration: per-boundary second moments and the rotation Q.

Implements sections 6 and 8 of ``knowledge/width_pruning.md``.

Three facts drive the design here, all of them measured:

* **Audio and prefix positions must be accumulated separately.** Prefix
  (style/text) positions are ~15% of tokens but carry attention-sink activations
  that dominate the energy sum. Pooled, boundary 24 shows a single eigendirection
  holding 98.6% of the energy and a meaningless retention of 0.9996; audio-only
  it is 62% and 0.9795. The loss is computed on audio positions only.
* **Each boundary's moment must be trace-normalized before summing.** Residual
  energy grows ~5400x from boundary 0 to 21 (and ~75,000x by boundary 27), so a
  raw sum is decided by a handful of deep boundaries: 0.935 -> 0.719 retention.
* **Do not center.** A centered basis needs a bias to add the mean back, and this
  architecture has no biases anywhere (``attention_bias: false``). The centered
  spectrum is computed too, but only as a diagnostic.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .manifest import to_device

AUDIO = "audio"
PREFIX = "prefix"


@dataclass
class BoundaryMoments:
    """Uncentered second-moment sums per residual boundary, per region.

    ``M[l] = sum_t x_t x_t^T`` and ``s[l] = sum_t x_t`` over positions in the
    region, accumulated in float64 on CPU. Memory is O(d^2) per boundary — never
    O(N*d) — so raw activations are never stored.
    """

    d: int
    num_boundaries: int
    M: Dict[str, torch.Tensor] = field(default_factory=dict)
    s: Dict[str, torch.Tensor] = field(default_factory=dict)
    n: Dict[str, torch.Tensor] = field(default_factory=dict)
    num_samples: int = 0
    ms_per_sample: float = 0.0
    fisher: bool = False

    @classmethod
    def zeros(cls, d: int, num_boundaries: int, fisher: bool = False):
        obj = cls(d=d, num_boundaries=num_boundaries, fisher=fisher)
        for reg in (AUDIO, PREFIX):
            obj.M[reg] = torch.zeros(num_boundaries, d, d, dtype=torch.float64)
            obj.s[reg] = torch.zeros(num_boundaries, d, dtype=torch.float64)
            obj.n[reg] = torch.zeros(num_boundaries, dtype=torch.float64)
        return obj

    def save(self, path: str) -> None:
        torch.save(
            {
                "d": self.d,
                "num_boundaries": self.num_boundaries,
                "M": self.M,
                "s": self.s,
                "n": self.n,
                "num_samples": self.num_samples,
                "ms_per_sample": self.ms_per_sample,
                "fisher": self.fisher,
            },
            path,
        )

    @classmethod
    def load(cls, path: str):
        blob = torch.load(path, map_location="cpu")
        return cls(**blob)


def _hidden_states(model, batch, output_hidden_states: bool = True):
    """Run the backbone and return (hidden_states, logits).

    ``model.llm`` is a ``Qwen3Model``: with ``output_hidden_states=True`` it
    returns ``num_hidden_layers + 1`` tensors where index 0 is post-embedding and
    the LAST is post-final-norm (HF applies ``self.norm`` before appending). That
    is exactly the boundary numbering used in ``knowledge/width_pruning.md``.
    """
    embeds = model._prepare_embed_inputs(batch["input_ids"], batch["audio_mask"])
    out = model.llm(
        inputs_embeds=embeds,
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    return out.hidden_states, out.last_hidden_state


def _audio_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    b, s, _ = hidden.shape
    flat = model.audio_heads(hidden)
    return flat.view(
        b, s, model.config.num_audio_codebook, model.config.audio_vocab_size
    ).permute(0, 2, 1, 3)


def _weighted_loss(model, audio_logits: torch.Tensor, labels: torch.Tensor):
    per_token = torch.nn.functional.cross_entropy(
        audio_logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=-100
    )
    valid = (labels != -100).float()
    layer_means = (per_token * valid).sum(dim=(0, 2)) / valid.sum(dim=(0, 2)).clamp(
        min=1.0
    )
    w = torch.tensor(
        model.normalized_audio_codebook_weights, device=audio_logits.device
    )
    return (layer_means * w).sum()


def _region_masks(audio_mask: torch.Tensor, valid: torch.Tensor):
    """Boolean ``[B, L]`` masks for the two regions, padding excluded.

    Padding must be dropped explicitly: the collator pads ``audio_mask`` with
    False, so ``~audio_mask`` would sweep every pad position into the prefix
    statistics.
    """
    return ((AUDIO, valid & audio_mask), (PREFIX, valid & ~audio_mask))


@torch.no_grad()
def _accumulate_plain(moments: BoundaryMoments, hidden_states, audio_mask, valid):
    for reg, mask in _region_masks(audio_mask, valid):
        count = int(mask.sum())
        if count == 0:
            continue
        # h is [B, L, d] and mask is [B, L], so h[mask] gathers every position in
        # the batch, not just the first row. [NB, n, d]; stacking lets one bmm
        # cover all boundaries at once.
        X = torch.stack([h[mask].float() for h in hidden_states])
        Mb = torch.bmm(X.transpose(1, 2), X)
        # MPS has no float64: move to CPU *before* casting, not after.
        moments.M[reg] += Mb.cpu().double()
        moments.s[reg] += X.sum(dim=1).cpu().double()
        moments.n[reg] += count


def _accumulate_fisher(moments: BoundaryMoments, hidden_states, audio_mask, valid, grads):
    """Output-sensitivity weighted accumulation (section 6.2).

    Plain PCA maximizes explained variance of the *activation*; what matters is
    explained variance of the *output*. The exact objective is not a quadratic
    form in the projector, so this uses the standard tractable surrogate: weight
    each position's outer product by the squared gradient norm at that position,
    ``M = sum_t ||dL/dx_t||^2 x_t x_t^T``. Analogous to what GPTQ/AWQ do for
    quantization and FWSVD for factorization.
    """
    for reg, mask in _region_masks(audio_mask, valid):
        count = int(mask.sum())
        if count == 0:
            continue
        acc_M = torch.zeros(
            moments.num_boundaries, moments.d, moments.d, dtype=torch.float64
        )
        acc_s = torch.zeros(moments.num_boundaries, moments.d, dtype=torch.float64)
        for b, (h, g) in enumerate(zip(hidden_states, grads)):
            x = h[mask].float()
            if g is None:
                w = torch.ones(x.shape[0], 1, device=x.device)
            else:
                w = g[mask].float().pow(2).sum(dim=-1, keepdim=True)
                w = w / w.mean().clamp(min=1e-12)  # keep the scale comparable
            xw = x * w
            acc_M[b] = (xw.T @ x).cpu().double()
            acc_s[b] = xw.sum(dim=0).cpu().double()
        moments.M[reg] += acc_M
        moments.s[reg] += acc_s
        moments.n[reg] += count


def calibrate(
    model,
    dataloader,
    device,
    fisher: bool = False,
    max_batches: int = 0,
    log_every: int = 20,
    verbose: bool = True,
) -> BoundaryMoments:
    """Accumulate per-boundary second moments over the calibration set.

    The calibration set must span the diffusion time axis: the processor draws
    ``mask_ratio ~ U(0, 1)`` per sample, so a single noise level never occurs.
    Note that training masks iid across codebooks while inference applies a
    layer penalty that forces low codebooks to resolve first — see section 9.
    """
    d = model.config.llm_config.hidden_size
    nb = model.config.llm_config.num_hidden_layers + 1
    moments = BoundaryMoments.zeros(d, nb, fisher=fisher)

    was_training = model.training
    model.eval()
    elapsed, seen = 0.0, 0

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        batch = to_device(batch, device)
        bsz = batch["input_ids"].shape[0]
        # attention_mask[b, 0, i, j] == valid[b, j]
        valid = batch["attention_mask"][:, 0, 0, :].bool()
        t0 = time.perf_counter()

        if not fisher:
            with torch.no_grad():
                hs, _ = _hidden_states(model, batch)
                _accumulate_plain(moments, hs, batch["audio_mask"], valid)
        else:
            hs, last = _hidden_states(model, batch)
            for h in hs:
                h.retain_grad()
            loss = _weighted_loss(model, _audio_logits(model, last), batch["labels"])
            model.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                _accumulate_fisher(
                    moments,
                    [h.detach() for h in hs],
                    batch["audio_mask"],
                    valid,
                    [h.grad for h in hs],
                )
            model.zero_grad(set_to_none=True)

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        seen += bsz
        if verbose and log_every and (i + 1) % log_every == 0:
            print(
                f"  calib {i + 1} batches / {seen} samples  "
                f"{1000 * elapsed / seen:.1f} ms/sample",
                flush=True,
            )

    moments.num_samples = seen
    moments.ms_per_sample = 1000 * elapsed / max(seen, 1)
    _check_rank(moments, d)
    if was_training:
        model.train()
    if verbose:
        na = int(moments.n[AUDIO][0])
        npre = int(moments.n[PREFIX][0])
        print(
            f"  calibrated on {seen} samples in {elapsed:.1f}s "
            f"({moments.ms_per_sample:.1f} ms/sample); "
            f"audio positions={na} prefix={npre} "
            f"(prefix share {npre / max(na + npre, 1):.3f})"
        )
    return moments


def _check_rank(moments: BoundaryMoments, d: int) -> None:
    """Refuse a calibration set that cannot determine the basis.

    ``M`` has rank at most the number of accumulated positions. With fewer than
    ``d`` audio positions the top-``k`` eigenvectors are arbitrary inside the
    null space, and retention reads a meaningless 1.0000 at every boundary
    because a rank-deficient moment is trivially captured. The resulting Q
    discards directions that merely happened to carry no energy on the sample,
    which destroys the model. Aim for >= 10x the width.
    """
    n_audio = float(moments.n[AUDIO][0])
    if n_audio < d:
        raise RuntimeError(
            f"calibration collected only {int(n_audio)} audio positions for a "
            f"{d}-wide stream: the second moment is rank-deficient and Q would "
            f"be arbitrary. Raise --calib-samples (>= {int(10 * d / 150)} "
            f"utterances gives ~10x coverage at this corpus's median length)."
        )
    if n_audio < 10 * d:
        print(
            f"  WARNING: only {int(n_audio)} audio positions for a {d}-wide "
            f"stream ({n_audio / d:.1f}x). The spectrum will be noisy; >=10x is "
            f"recommended."
        )


# --------------------------------------------------------------------------
# Building Q from the moments
# --------------------------------------------------------------------------


def _eigh_desc(M: torch.Tensor):
    w, V = torch.linalg.eigh(M)
    return w.flip(0), V.flip(1)


def compute_rotation(
    moments: BoundaryMoments,
    region: str = AUDIO,
    trace_normalize: bool = True,
    boundaries: Optional[List[int]] = None,
) -> torch.Tensor:
    """Return the full orthogonal ``Q`` (``[d, d]``, columns ordered by energy).

    ``trace_normalize`` is not optional in practice — leaving it off costs 21
    points of worst-case retention (0.935 -> 0.719). It is a flag only so the
    failure can be reproduced.
    """
    M = moments.M[region]
    idx = boundaries if boundaries is not None else range(moments.num_boundaries)
    total = torch.zeros(moments.d, moments.d, dtype=torch.float64)
    for l in idx:
        Ml = M[l]
        if trace_normalize:
            tr = torch.diagonal(Ml).sum().clamp(min=1e-30)
            Ml = Ml / tr
        total += Ml
    _, V = _eigh_desc(total)
    return V


def channel_selection(
    moments: BoundaryMoments,
    k: int,
    region: str = AUDIO,
    trace_normalize: bool = True,
) -> torch.Tensor:
    """Axis-aligned alternative to rotation (section 6.3).

    Returns a permutation matrix whose first ``k`` columns select the ``k``
    highest-power coordinates, so it drops into the same folding code as a real
    rotation. Worth testing because this model has extreme outlier-channel
    structure: participation ratio 4-8 of 1024 in the deep layers. Those may be
    massive-activation channels that dominate energy while carrying little
    information, which is precisely what Fisher weighting detects — judge on
    post-recovery loss, not on the concentration statistic.
    """
    M = moments.M[region]
    power = torch.zeros(moments.d, dtype=torch.float64)
    for l in range(moments.num_boundaries):
        d_l = torch.diagonal(M[l])
        if trace_normalize:
            d_l = d_l / d_l.sum().clamp(min=1e-30)
        power += d_l
    order = torch.argsort(power, descending=True)
    Q = torch.zeros(moments.d, moments.d, dtype=torch.float64)
    Q[order, torch.arange(moments.d)] = 1.0
    return Q


def retention(moments: BoundaryMoments, Q: torch.Tensor, k: int, region: str = AUDIO):
    """Per-boundary energy retained by ``Q[:, :k]``: ``tr(Qk^T M Qk) / tr(M)``."""
    Qk = Q[:, :k]
    out = []
    for l in range(moments.num_boundaries):
        Ml = moments.M[region][l]
        tr = torch.diagonal(Ml).sum().clamp(min=1e-30)
        out.append(float(torch.diagonal(Qk.T @ Ml @ Qk).sum() / tr))
    return np.array(out)


def diagnostics(moments: BoundaryMoments, k: int, region: str = AUDIO) -> dict:
    """Per-boundary spectrum diagnostics. Cheap; run it every rung.

    ``mean_share`` is the fraction of uncentered energy in the mean vector alone.
    In the deep layers of the 1024-wide teacher over half the residual energy is
    a constant offset (0.97 at the final norm), so the information-bearing
    variance is markedly higher-dimensional than the uncentered spectrum implies.

    ``participation_ratio`` is ``(sum d)^2 / sum d^2`` over per-channel mean
    squares: 1 means all power in one channel, ``d`` means uniform.
    """
    d, nb = moments.d, moments.num_boundaries
    own, centered, mean_share, pr = [], [], [], []
    for l in range(nb):
        Ml = moments.M[region][l]
        n = max(float(moments.n[region][l]), 1.0)
        w, _ = _eigh_desc(Ml)
        own.append(float(w[:k].sum() / w.sum().clamp(min=1e-30)))
        mu = moments.s[region][l] / n
        tr = torch.diagonal(Ml).sum().clamp(min=1e-30)
        mean_share.append(float(n * (mu @ mu) / tr))
        C = Ml / n - torch.outer(mu, mu)
        wc, _ = _eigh_desc(C)
        centered.append(float(wc[:k].sum() / wc.sum().clamp(min=1e-30)))
        diag = torch.diagonal(Ml) / n
        pr.append(float(diag.sum() ** 2 / (diag**2).sum().clamp(min=1e-30)))
    return {
        "retention_own_basis": np.array(own),
        "centered_retention": np.array(centered),
        "mean_share": np.array(mean_share),
        "participation_ratio": np.array(pr),
        "trace": np.array(
            [float(torch.diagonal(moments.M[region][l]).sum()) for l in range(nb)]
        ),
    }
