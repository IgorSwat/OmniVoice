# Width-pruning scripts

Progressive width pruning of the OmniVoice Qwen3 backbone: shrink `hidden_size`
one rung at a time, repair each cut in closed form, then recover with knowledge
distillation. Theory, algebra and the measured numbers for this checkpoint live
in [`knowledge/width_pruning.md`](../knowledge/width_pruning.md).

## Layout

| File | Role |
|---|---|
| `progressive_width_prune.py` | Entry point. Runs the whole ladder. |
| `width_pruning/manifest.py` | CSV-manifest dataset, length-grouped batching. |
| `width_pruning/calibration.py` | Per-boundary second moments; builds `Q`. |
| `width_pruning/surgery.py` | Gain folding, `Q` folding, truncation, round-trip test. |
| `width_pruning/repair.py` | Closed-form least-squares repair. |
| `width_pruning/distill.py` | KD recovery loop and the dev-loss gate. |

## Usage

Full ladder on a GPU:

```bash
python scripts/progressive_width_prune.py \
    --widths 896 800 704 \
    --out-dir runs/ladder_v1 \
    --steps 4000 --batch-tokens 16384 --device cuda
```

Cheap sweep with no training — surgery and repair only, which is the ~seconds-per-
candidate proxy for choosing `k`:

```bash
python scripts/progressive_width_prune.py \
    --widths 832 --skip-recovery --out-dir runs/sweep_832 --device mps
```

Basis A/B (section 6.2/6.3 of the knowledge doc — run all three and judge on
post-recovery loss, not on the retention statistic):

```bash
for b in pca fisher selection; do
  python scripts/progressive_width_prune.py --widths 704 --basis $b \
      --skip-recovery --out-dir runs/basis_$b --device mps
done
```

Each rung writes `runs/<name>/rung_<k>/` containing a stock HuggingFace
checkpoint, `moments.pt` (the calibration second moments), and `rotation.pt`
(`Q`, `k`, and the cumulative teacher→student projection). A single
`report.json` at the top level carries retention curves, repair errors,
post-surgery loss, recovered loss and gate outcomes for every rung.

## Design decisions worth knowing

**Data comes straight from the CSV manifests.** The training pipeline in
`omnivoice.training.builder` reads WebDataset tar shards; this reads
`data/*.csv` plus the pre-tokenized codec `.npy` files directly, so no shard
conversion is needed. Sample lengths are cached as `<manifest>.lengths.npy`.

**Attention is bidirectional.** Batches come from `PaddingDataCollator`, whose
4D `[B, 1, L, L]` mask stops HuggingFace from adding a causal mask. Passing a 2D
mask or `None` would silently make this a causal model and every number would be
wrong.

**Recovery always distills from the original teacher**, never from the previous
rung's student, so errors do not compound down the ladder. Repair, by contrast,
targets the immediately-pre-cut model, because it is a local reconstruction
problem.

**`Q` is re-fit on the current student at every rung.** Importance is only
locally valid: a `Q` fit to the 1024-wide model does not describe the residual
stream of an 896-wide one.

**One global `Q`, no adapters.** The student stays a stock `Qwen3Model` at the
new `hidden_size`, loadable by HuggingFace with no custom code. `head_dim` stays
128 — it is explicit in `Qwen3Config` and decoupled from `hidden_size`, so RoPE
and `q_norm`/`k_norm` are untouched.

## Two places this deliberately departs from the knowledge doc

1. **The RMSNorm dimension rescale is `sqrt(d/k)`, not `sqrt(k/d)`.** Section 3
   states the factor inverted. `rms` averages over dimensions, so at fixed vector
   norm the normalizer grows by `sqrt(d/k)` and the normalized output shrinks by
   `sqrt(k/d)`; read weights must therefore be scaled *up* by `sqrt(d/k)`. The
   exact factor is `sqrt(d/k * r_l)` with `r_l` the retention at that boundary;
   the `sqrt(r_l)` term is a ~1% correction the repair absorbs.

2. **Codebook weighting is not applied to the closed-form repair,** because
   there it is provably a no-op. `sum_c w_c ||y_c - W_c h||^2` decouples over
   codebooks — each owns a disjoint block of output rows and the design matrix is
   shared — so a positive `w_c` rescales each subproblem without moving its
   argmin. The weights are applied in the distillation loss, where the objective
   is genuinely joint and they do bite.

## Caveats

- **`--hidden-weight` defaults to 0.** The teacher→student stream map is exact at
  initialization and degrades as recovery training drifts the student off the
  rotated basis. It is a warm-start regularizer, not a stable objective.
- **Use `--repair-mode sequential`.** Measured at k=704 on the dev set,
  post-surgery loss is 5.85 sequential vs 7.46 joint against a teacher's 4.58 —
  sequential more than halves the excess damage (+27.8% vs +62.9%), because each
  layer is refit against inputs its predecessors have already been corrected
  for. It costs `num_layers + 1` passes (~11 min on MPS, far less on a GPU) and
  holds ~130 MB; `joint` is one pass but holds ~4 GB and ignores propagation.
  Use `joint` only for quick plumbing checks.
- **`--calib-samples 192` is a floor, not a suggestion.** The second moment has
  rank at most the number of accumulated positions, so below `d` audio positions
  the top-`k` eigenvectors are arbitrary inside the null space — and retention
  reads a meaningless `1.0000` at every boundary, because a rank-deficient
  moment is trivially captured. Calibration now hard-errors below `d` positions
  and warns below `10x`. A retention column of straight `1.0000` always means
  the calibration set is too small, never that the cut is free.
- **The dev-loss gate is necessary, not sufficient.** Loss can look fine while
  prosody degrades. Confirm shortlisted rungs with the generation metrics in
  `omnivoice/eval/` (`wer`, `speaker_similarity`, `mos`).
- **Calibration masks are training masks, not inference masks.** Training masks
  iid uniformly across codebooks; inference applies a layer penalty that forces
  low codebooks to resolve first, so the surviving masked set is skewed toward
  high codebooks in a way the processor never produces. For the most faithful
  calibration, harvest states from real generation trajectories.
- **Retention is a capacity signal, not a damage estimate.** Under
  prune-then-retrain it bounds *initialization* damage only. Pick `k` from the
  decision surface, not from a 99% threshold.
