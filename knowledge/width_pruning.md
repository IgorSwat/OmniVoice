# Width pruning of the OmniVoice residual stream

How to shrink the Qwen3 backbone's `hidden_size` from **1024 → 704** by rotating the residual
stream onto its dominant subspace and truncating, with closed-form repair.

Self-contained: theory, algebra, the design decisions, measured numbers for this checkpoint, and
implementation gotchas.

**Provenance.** Measurements were taken against the HF checkpoint `k2-fsa/OmniVoice`
(snapshot `c5fdb5cc`, 612.58M params, 313 tensors) using a copy of this repo at commit-equivalent
state *older than `08be0b4`*. Architecture and checkpoint are identical; **line numbers cited below
are for this checkout**, but prefer the symbol names — they are stable, line numbers are not.

Markers: ✅ measured · 📊 computed from verified values · ⚠️ unverified / from memory

---

## Table of contents

1. [What is being pruned](#1-what-is-being-pruned)
2. [The core trick: rotation is exactly free](#2-the-core-trick-rotation-is-exactly-free)
3. [The RMSNorm precondition](#3-the-rmsnorm-precondition)
4. [Truncation — where loss enters](#4-truncation--where-loss-enters)
5. [Least-squares repair](#5-least-squares-repair)
6. [Choosing Q](#6-choosing-q)
7. [One Q, per-layer Q, and adapters](#7-one-q-per-layer-q-and-adapters)
8. [Measured results for this checkpoint](#8-measured-results-for-this-checkpoint)
9. [Implementation checklist](#9-implementation-checklist)
10. [What this method does NOT cover](#10-what-this-method-does-not-cover)
11. [How it fits a progressive pruning process](#11-how-it-fits-a-progressive-pruning-process)

---

## 1. What is being pruned

A pre-norm transformer is best read not as a stack of functions but as a **shared accumulating
buffer** that every layer reads from and writes into:

```
x₀ ──┬──────────────────────────┬──► x₁ ──┬───────────┬──► x₂ ──► … ──► x₂₈
     │                          │         │           │
     └─► RMSNorm ─► Attn ───────┘         └─► MLP ────┘
                              (added back into the stream)
```

Each layer computes `x_{l+1} = x_l + f(x_l)`; the `x_l` term passes through untouched on the skip
connection. In this model that buffer is **1024-dimensional at every token position**, and it is the
*same* buffer from `audio_embeddings` all the way to `audio_heads`. Every module agrees on what
those 1024 coordinates mean.

**Width pruning makes that buffer 704-dimensional.**

It is possible because the buffer does not use its dimensions evenly. Treated as a point cloud in
ℝ¹⁰²⁴, the activations concentrate: ~704 directions capture ~98% of the energy (§8). The stream
lives on a lower-dimensional subspace than the one allocated to it.

**Relevant architecture** ✅ (`llm_config` of the checkpoint):

| Field | Value | Note |
|---|---|---|
| `hidden_size` | 1024 | ← the thing being pruned |
| `num_hidden_layers` | 28 | 29 residual boundaries (0 = post-embedding, 28 = post-final-norm) |
| `head_dim` | **128, explicit** | decoupled from `hidden_size`, so RoPE and `q_norm`/`k_norm` are untouched by width changes |
| `intermediate_size` | 3072 | a *different* axis — see §10 |
| `attention_bias` | **false** | no biases anywhere; this is load-bearing (§3, §6.1) |
| `rms_norm_eps` | 1e-6 | |

---

## 2. The core trick: rotation is exactly free

The naive way to shrink is to delete 320 of the 1024 coordinates. That is bad: the point cloud is a
tilted ellipsoid, and its principal directions are not aligned with the coordinate axes.

So **rotate first, then truncate**. The rotation itself is free, and here is why.

Take any orthogonal `Q` (1024×1024, `QᵀQ = I`). Decide the buffer will now hold `x' = Qᵀx`. Then:

- A module that **reads** the stream, `y = Wx`: since `x = Qx'`, we get `y = (WQ)x'`.
  → fold `Q` into the **input side** of the read weights.
- A module that **writes** to the stream, producing `y`: it must now emit `Qᵀy = (QᵀW)h`.
  → fold `Qᵀ` into the **output side** of the write weights.

Using PyTorch's convention (`nn.Linear(in, out).weight` has shape `[out, in]`):

| Role | Module | Transform |
|---|---|---|
| **write** | `llm.embed_tokens` (`[vocab, 1024]`) | `E ← E Q` |
| **write** | `audio_embeddings` (`[8×1025, 1024]`) | `E ← E Q` |
| **read** | `q_proj`, `k_proj`, `v_proj` (`[·, 1024]`) | `W ← W Q` |
| **write** | `o_proj` (`[1024, ·]`) | `W ← Qᵀ W` |
| **read** | `gate_proj`, `up_proj` (`[3072, 1024]`) | `W ← W Q` |
| **write** | `down_proj` (`[1024, 3072]`) | `W ← Qᵀ W` |
| **read** | `audio_heads` (`[8×1025, 1024]`) | `W ← W Q` |

> Convention-independent statement of the rule: **read-side gets `Q` on the stream-facing side,
> write-side gets `Qᵀ` on the stream-facing side.** Derive the concrete transpose from your own
> weight layout rather than copying the table blindly.

**Why the residual connection survives it:** `Qᵀ(a + b) = Qᵀa + Qᵀb`. Rotation distributes over the
skip addition, so the pass-through term needs no special handling.

**Result: a network with different weights computing exactly the same function**, up to float error.
Nothing has been pruned. You have only chosen better axes. This is worth verifying numerically as a
unit test before truncating anything — a full-rank `Q` round-trip must be a no-op.

---

## 3. The RMSNorm precondition

`RMSNorm(x) = (x / rms(x)) · g`, where `rms(x) = ‖x‖/√d`.

- The `x / rms(x)` part **commutes** with `Q`, because `‖Qᵀx‖ = ‖x‖` for orthogonal `Q`.
- The learnable gain `g` is **elementwise**, i.e. `diag(g)`, which does **not** commute with a
  rotation.

**Therefore: fold each RMSNorm's gain `g` into the following linear layer first**, leaving pure RMS
normalization. Only then is `Q` exactly commuting.

Qwen3 satisfies the remaining preconditions natively — pre-norm, RMSNorm, and **no biases anywhere**
(`attention_bias: false`, no MLP bias, RMSNorm has no bias). A bias would break the algebra, since
`W(Qx') + b` needs `b` rotated too, and biases sitting on the residual stream would have to be
rotated consistently. **No conversion step is needed for this architecture.**

**Dimension rescale.** `rms` averages over dimensions, so changing `d = 1024 → d' = 704` changes the
normalizer by `√(d/d')` for a vector of the same norm. Fold a compensating `√(d'/d)` into the
following weights.

---

## 4. Truncation — where loss enters

Keep only the first `k = 704` columns of `Q`:

- read weights: `W · Q[:, :k]` → `[out, 704]`
- write weights: `Q[:, :k]ᵀ · W` → `[704, in]`

The buffer is now 704-wide, and the model is a **stock `Qwen3Model` with `hidden_size: 704`** —
provided a single global `Q` was used (§7).

**This is the only lossy step.** Its cost is exactly the energy in the 320 discarded directions.

**How to read a retention number.** If the top-`k` subspace retains fraction `r` of `Σ‖x‖²`, then the
mean squared relative error is `1 − r`, so the **RMS relative perturbation is `√(1 − r)`**:

| retention `r` | RMS perturbation |
|---|---|
| 0.9955 | 6.7% |
| 0.9795 | 14.3% |
| 0.9356 | 25.4% |

⚠️ A "99% gate" sounds like a rounding error and is not — it is a ~10% perturbation of every
activation vector. But see §11: under prune-then-retrain this is *initialization* damage, and must
not be read as compounding across layers and diffusion steps.

---

## 5. Least-squares repair

After truncation the outputs are wrong by the projection residual. Refit the **write-side**
projections to best reproduce the teacher's output given the student's inputs:

```
min_W  ‖ W_teacher · h_teacher  −  W · h_student ‖² ,  ridge-regularized
```

Closed-form (normal equations), no gradients, minutes. Accumulate `XᵀX` and `XᵀY` during the
calibration pass — never store raw activations.

**Weight the objective by the codebook weights.** The training loss is a weighted sum of per-codebook
means with `audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]`, normalized in
`OmniVoice.__init__` ([`omnivoice/models/omnivoice.py:316`](../omnivoice/models/omnivoice.py#L316)).
Codebook 0 carries **4× the weight** of codebook 7; repair that ignores this optimizes the wrong
thing.

This closed-form repair is what makes width pruning cheap to *evaluate*: a candidate can be cut,
repaired, and scored in seconds. Depth pruning has no such repair — deleting a block removes an
entire nonlinear function, and only fine-tuning recovers it.

---

## 6. Choosing Q

### 6.1 Use the uncentered second moment `E[xxᵀ]`

Accumulate `M = Σ x xᵀ` over calibration positions and take the top eigenvectors.

**Do not center.** A centered covariance implicitly requires a bias term to add the mean back at
inference, and **this architecture has no biases** to absorb one. Centering silently degrades
everything downstream.

⚠️ **But the uncentered spectrum is a misleading diagnostic**, because a large share of the energy is
a near-constant offset (§8): the top eigendirection is partly "the mean," and retaining it tells you
little. Compute the centered spectrum too, purely as a *diagnostic*, to see how much of your
retention is information versus constant.

### 6.2 Optional: Fisher / output-sensitivity weighting

Plain PCA maximizes explained variance of the **activation**. What you want is explained variance of
the **output**. A high-variance direction with no downstream effect wastes a dimension; a
low-variance direction the output is sensitive to must be kept.

Weight the second moment by `∂L/∂x` instead of accumulating raw `xxᵀ`. Costs one backward pass over
the calibration set (measured at 2.7× a forward — seconds at this scale, §8). Analogous to what
GPTQ/AWQ do for quantization and FWSVD for factorization.

**Run plain and Fisher-weighted both and compare.** Cheap A/B.

### 6.3 Optional: no rotation at all (channel selection)

The simplest alternative is axis-aligned **selection** — keep 704 of the 1024 coordinates, no `Q`
anywhere. The architecture stays literally identical, embedding rows just get sliced, and the entire
`Q`-folding apparatus disappears.

Normally much worse than PCA, but this model has strong outlier-channel structure (§8), so selection
may be more competitive than expected. ⚠️ Do not conclude from the concentration statistic alone —
those channels may be massive-activation / attention-sink channels that dominate *energy* while
carrying little *information*, which is exactly the failure Fisher weighting detects. Test all three
(PCA / Fisher-PCA / selection) and judge on post-recovery loss.

---

## 7. One Q, per-layer Q, and adapters

The residual stream is *shared*, so every module must agree on the basis. But **the ideal `Q` differs
with depth** — the activation distribution at boundary 2 is not the one at boundary 24 (§8).

### One global Q

Fold the same `Q` everywhere. The student remains a **stock Qwen3** loadable by HuggingFace with
`hidden_size: 704`. Zero extra parameters, zero architecture change. This is the recommended default
and what the compression plan assumes.

### Per-layer Q (SliceGPT)

Give each layer its own `Q_l`. Better fit — but layer `l` writes the stream in basis `Q_l` while
layer `l+1` expects `Q_{l+1}`, so the bases must be reconciled.

**You cannot hide that conversion inside the layer weights, because the skip connection bypasses
them** — the `x_l` term passes through untouched. So an explicit matrix must sit on the residual
path itself. **That matrix is the adapter:**

```
one Q:          x ─────────────────────────────► x     (nothing needed)

per-layer Q:    x ──► [  Q_lᵀ · Q_{l+1}  ] ─────► x     (an ADAPTER)
                      └──── 704 × 704 ────┘
```

Costs: ~0.5M params each (704² ≈ 495k) 📊, extra compute on **every** residual connection, and — the
real objection — the model is no longer stock Qwen3.

### Blocks — the middle ground

Group **contiguous** layers. All layers in a block share one basis, so no adapters inside a block;
one adapter at each junction *between* blocks.

> **N blocks ⇒ N − 1 adapters.**

This is a genuine dial between "0 adapters, worst fit" and "27 adapters, best fit", and the
measurements in §8 show most of the gap closes with 1–2 adapters.

---

## 8. Measured results for this checkpoint ✅

Calibration: 192 utterances (~19.7 min, all English, median 6.1 s) from a held-out dev set,
pre-tokenized codec `.npy`, teacher fp16 on Apple M4 Pro / MPS, SDPA attention.

**Cost — calibration is a non-issue:**

| Pass | ms/sample | 192 samples |
|---|---|---|
| Forward + second-moment accumulation | 76 | **15 s** |
| \+ backward (for Fisher weighting) | 307 | **59 s** |

Batching barely helps (99.8 → 76.1 ms/sample from B=1 to B=4, flat after) — MPS saturates at B=1.

### 8.1 Retention, audio positions only

⚠️ **Accumulate audio-region and prefix statistics separately.** Prefix (style/text) positions are
only 16.7% of tokens but carry huge attention-sink activations; pooled, they dominate the energy sum
and flatter the result badly. Pooled over all positions, boundary 24 shows participation ratio 1.94
and a *single* eigendirection holding 98.7% of energy — the pooled spectrum largely certifies that
the sink direction survives, which is not the question. **The loss is computed on audio positions
only** (`labels = -100` on style and text, set in `OmniVoiceSampleProcessor.__call__`).

Uncentered `E[xxᵀ]`, audio positions (n = 29,536), energy in top-704 of 1024, **per-boundary optimal
basis**:

| Boundaries | retention @ k=704 |
|---|---|
| 0–18 | 0.9932 – 0.9998 |
| **19–27** | **0.9795 – 0.9905** |
| 28 (post final norm) | 0.9995 |

### 8.2 The global-Q constraint costs ~4 points

Per-boundary figures each use their *own* optimal basis — 29 different ones. A single global `Q` must
serve all of them:

| Basis, k=704 | min retention | median |
|---|---|---|
| Per-boundary optimal (29 bases) | 0.9795 | 0.9955 |
| **Single global `Q`, trace-normalized** | **0.9356** | **0.9811** |
| Single global `Q`, **raw sum** | **0.7192** | 0.9497 |

⚠️ **Trace-normalize each boundary's moment before summing them.** Total residual energy grows
**5,218×** from boundary 0 to boundary 21, so an unweighted `Σ M_l` is dominated by a handful of deep
boundaries and collapses to **0.7192**. Same data, same method, **22 points worse**. This is the
naive implementation and it is badly wrong.

### 8.3 Blocks vs. width — the decision surface

| k=704 | min retention | adapters |
|---|---|---|
| 1 block (global `Q`) | 0.9356 | 0 |
| 2 blocks, split @21 | 0.9545 | 1 |
| 3 blocks, split @(21, 25) | 0.9635 | 2 |
| 28 blocks (per-layer) | 0.9795 | 27 |

| k=832 | min retention | adapters |
|---|---|---|
| 1 block (global `Q`) | **0.9656** | 0 |
| 2 blocks, split @21 | 0.9762 | 1 |
| 3 blocks, split @(21, 25) | 0.9810 | 2 |

**Two adapters recover ~64% of the global-`Q` gap.** Alternatively `k=832` with one global `Q`
(0.9656) beats 3 blocks at `k=704` (0.9635) with zero architecture change.

The optimal splits at **21 and 25** are not arbitrary — they coincide with where Block Influence
spikes (layer 21 = 0.373, layer 23 = 0.225 against a 0.017–0.033 plateau over layers 1–15), i.e. two
independent measurements agree on the same structural seam.

**`d`, block count, and recovery budget trade against each other, and each variant costs ~15 s to
evaluate.** Pick from these curves; do not treat any single retention threshold as a gate.

### 8.4 Constant-offset structure

`meanShare` = fraction of uncentered audio-region energy lying in the mean vector alone:

| Boundary | meanShare | centered retention @ k=704 |
|---|---|---|
| 7 | 0.38 | 0.9949 |
| 14 | 0.42 | 0.9933 |
| 20 | 0.53 | 0.9799 |
| **24** | **0.61** | **0.9481** |
| 27 | 0.52 | 0.9622 |
| 28 | **0.97** | 0.9827 |

In the deep layers **over half the residual energy is a constant offset**; at the final norm output,
97% is. The information-bearing variance is markedly higher-dimensional than the uncentered spectrum
suggests.

**Outlier-channel concentration**, as participation ratio `PR = 1024 / E[z²]` where
`z = dᵢ / mean(d)` and `d` is per-channel mean square (**baseline for uniform power is 1**):

| Region | boundary 0 | deep layers (19–27) |
|---|---|---|
| pooled (prefix + audio) | 131 | 1.9 – 3.0 |
| **audio only** | 128 | **4.3 – 8.4** |

Roughly 4–8 effective channels of 1024 carry the power in the deep layers — extreme, and the reason
§6.3 is worth testing.

### 8.5 Verdict

**Width pruning to 704 is expected to succeed** with repair plus recovery training. The operative
worst case is 0.9356 (global `Q`), ≈25% RMS perturbation at initialization — before repair and
before any training. Damage is localized to a contiguous deep band (boundaries 19–27) rather than
diffuse, which is the easier case to recover. ⚠️ For calibration, SliceGPT is recalled (unverified)
to slice 25–30% with *no* recovery fine-tuning at modest cost; here there is both repair and a
recovery loop.

Set `d` from the curves in §8.3, not from a 99% threshold.

---

## 9. Implementation checklist

**Order of operations**

1. Fold every RMSNorm gain `g` into the following linear layer (§3).
2. Run the calibration pass, accumulating `M = Σxxᵀ` **per residual boundary**, separately for
   audio and prefix positions.
3. Trace-normalize each `M_l`, then sum within each block; eigendecompose; take top-`k`.
4. Verify the full-rank round-trip is a no-op **before truncating**.
5. Fold `Q` into all read/write weights (§2), including the `√(d'/d)` rescale (§3).
6. Truncate to `k` columns.
7. Least-squares repair of the write-side projections, codebook-weighted (§5).
8. Emit a stock `Qwen3Config` with `hidden_size: k` — leave `head_dim: 128` alone.

**Gotchas**

- ⚠️ **Trace-normalize before summing boundaries** — 22-point difference (§8.2).
- ⚠️ **Separate audio from prefix positions** — changes the headline conclusion (§8.1).
- ⚠️ **Fold RMSNorm gains first**, or `Q` does not commute (§3).
- ⚠️ **Weight repair by `audio_codebook_weights`** `[8,8,6,6,4,4,2,2]` (§5).
- Accumulate in **second-moment form**: memory is `O(d²)` (4 MB per boundary in fp32), not `O(N·d)`.
  Never store raw activations.
- **MPS does not support float64.** Move to CPU before casting: `(x.T @ x).cpu().double()`, not
  `.double().cpu()`.
- `head_dim` is **explicit** in `Qwen3Config` and decoupled from `hidden_size`, so RoPE and
  `q_norm`/`k_norm` (shape `[head_dim]`) are untouched by any width change. Attention projections
  simply become e.g. 1536-wide against a 704-wide residual — legal and intended.
- Calibration must span the diffusion time axis: `mask_ratio ~ U(*mask_ratio_range)`
  ([`processor.py:89`](../omnivoice/data/processor.py#L89)),
  `token_mask = torch.rand(...) < mask_ratio` ([`processor.py:142`](../omnivoice/data/processor.py#L142)).
  A single noise level yields a biased subspace.
- ⚠️ **Training masks ≠ inference masks.** Training masks iid uniformly across codebooks, but
  inference applies `scores - layer_ids * layer_penalty_factor`
  ([`omnivoice.py:1407`](../omnivoice/models/omnivoice.py#L1407), factor 5.0), forcing low codebooks
  to resolve first. At a given step the surviving masked set is skewed toward *high* codebooks — a
  composition the training processor never produces. For the most faithful calibration, harvest
  states from actual teacher generation trajectories.

---

## 10. What this method does NOT cover

**The FFN inner dimension (3072 → 1536) is a different axis.** `gate`/`up` project 1024→3072, SiLU
fires elementwise, `down` projects back. **Rotation is illegal there** — SiLU is elementwise and
`silu(Qh) ≠ Q·silu(h)`. Only axis-aligned **selection** works: pick 1536 of 3072 neurons.
`gate` and `up` must be pruned at **identical indices** (they are multiplied elementwise).
Score by `E[|hᵢ|] · ‖W_down[:, i]‖`, or better, orthogonal matching pursuit against the
post-refit residual.

**Attention heads** are also selection, not rotation: prune whole GQA groups (8 → 6), never merge KV
heads — attention's `S²` cost scales with query heads `H`, not `Kv`, and this model has no KV cache
(every diffusion step recomputes the full bidirectional forward), so GQA's usual payoff is absent.

**Depth pruning** is a wholly separate problem with **no closed-form repair**.

Neither the FFN spectrum nor per-head contribution norms have been measured yet; both come from the
same calibration pass at negligible extra cost.

---

## 11. How it fits a progressive pruning process

The intended process is **not** one-shot surgery. Each cut is followed by a short KD recovery, and
the target is quality *after* recovery. That changes how every number here must be read:

| Metric | Naive one-shot reading | Correct reading |
|---|---|---|
| Retention @ k | damage estimate; 99% gate | **capacity signal** — is `k` above the task's intrinsic dimension? A loose *upper* bound |
| Post-surgery loss | the decisive number | **weak predictor** of final quality; a proxy whose correlation with post-recovery loss must be verified, not assumed |
| RMS perturbation | compounds over 20 layers × 16 diffusion steps | applies to a *frozen* projection only; a retrained student is optimized to be self-consistent at the new width |

Practical consequences:

- **Sweep `k` and block count cheaply (~15 s each), then confirm the shortlist with recovery runs.**
- Establish **once** whether post-surgery loss ranks variants the same as post-recovery loss. If it
  does, sweep on the proxy. If not, every verdict needs a recovery run.
- **Re-fit `Q` after each round.** Importance is only locally valid; once you cut and repair, the
  activation distribution shifts. A `Q` fit to the 28-layer model does not describe the stream of a
  20-layer model — so if depth is also being cut, **cut depth first within each round.**
- Suggested ladder: `1024 → 896 → 800 → 704`, interleaved with depth, re-profiling at each stage.
- ⚠️ **Calibration here was English-only** (all 220 dev rows `language_id: en`), so these spectra are
  the teacher's *English* activation spectrum. The 646→10 language cut cannot be invoked to discount
  them directly. The real headroom is that a multilingual-trained teacher represents English
  non-compactly *for English*; a specialized student should learn a tighter code — genuine slack,
  but unquantified.
