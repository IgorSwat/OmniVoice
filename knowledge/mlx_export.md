# Exporting OmniVoice to ExecuTorch (MLX, CoreML, XNNPACK)

How to get `models/final` running under ExecuTorch on Apple silicon, which backend wins and why,
and the one number that governs every optimisation decision on the MLX delegate.

**Provenance.** All measurements are on an **Apple M4 Pro** (14 CPU cores, 20 GPU cores) against
`models/final` (400.0M params, vocab-pruned, 21 layers, hidden 1024), using ExecuTorch **1.4.1**
(`../executorch`, editable install, commit `091f752`), torch 2.13.0, coremltools 9.0.
Numbers are the **backbone only** — no codec decode, no sampler overhead, no Python loop — so they
are *not* directly comparable to the end-to-end RTF column in
[`distilation_plan.md` §11.10](distilation_plan.md).

Markers: ✅ measured · 📊 computed from verified values · ⚠️ unverified

---

## 1. The one-paragraph version

MLX is the default backend. The exported unit is the **prefix-cached split** — a prefix encoder run
once and a target step run per diffusion call — both with dynamic shapes in one graph each. The step
is **bound by graph interpretation, not arithmetic** (§5), which inverts the usual intuitions:
quantisation buys nothing, fp32 is nearly free, and the only lever that matters is node count. The
recommended configuration reaches **20.9 ms/step at 99.0% argmax agreement** — simultaneously faster
*and* more accurate than a naive fp16 export. ✅

---

## 2. Environment: use the executorch venv, do not install into this one

ExecuTorch lives in `../executorch/.venv` and already has everything. **Do not run
`install_executorch.sh`** — it is a long source build, and the packages are already present.

`omnivoice` is not installed there, and that venv's transformers (5.0.0rc1) is too old to import
`HiggsAudioV2TokenizerModel`. Bridge the two instead — nothing needs modifying:

```bash
OV_SP=$(python -c "import site;print(site.getsitepackages()[0])")
PYTHONPATH="/path/to/OmniVoice:$OV_SP" ../executorch/.venv/bin/python script.py
```

Both venvs are Python 3.12 on torch 2.13.0, so this is safe. **Path order matters**: the repo must
come *before* site-packages, or `omnivoice` resolves to an older installed copy whose
`from_pretrained` has a different signature (`train=` vs `train_mode=`). ✅

Load the model with `train=True` to skip the 200M audio codec, which is fetched separately and is
not part of the exported graph.

---

## 3. What gets exported

`forward()` with `document_ids=None` never touches `create_block_mask`, so the flex-attention path
is skipped and the graph is plain SDPA. ⚠️ Do not pass `document_ids` at export.

Two graphs, mirroring [`scripts/eval/prefix_cache.py`](../scripts/eval/prefix_cache.py) but with the
cache carried as plain tensors so both halves are exportable:

| graph | inputs | outputs |
|---|---|---|
| prefix encoder | `input_ids [1,8,P]`, `audio_mask [1,P]` | 42 K/V tensors `[1,8,P,128]` |
| target step | `input_ids [1,8,T]`, `audio_mask [1,T]`, 42 K/V | logits `[1,8,T,1025]` |

The masks are **all-attend** in both graphs — blocking is implicit in the split, since the prefix
graph only sees prefix positions. Verified exact against a full-sequence forward under the stage-2
blocked mask: max diff 1.37e-04, **100% argmax agreement**. ✅

Dynamic `P` and `T` (range 16–2048) cost nothing on MLX — measured identical to static, and correct
at every shape tried from T=60 to T=400. ✅

---

## 4. Backend comparison

Prefix-cached, arm e (8 steps + polish, 9 target calls), P=171 T=126, fp16. ✅

| backend | prefix ms | step ms | arm e | note |
|---|---|---|---|---|
| CoreML static, ANE | 17.1 | 13.6 | 36.1×RT | fastest, but static shapes only |
| **MLX dynamic** | 28.8 | 22.6 | 21.7×RT | **default** |
| CoreML enumerated, CPU+GPU | 28.8 | 24.9 | 20.0×RT | ANE lost |
| PyTorch MPS fp16 | 36.8 | 30.7 | 16.1×RT | reference |
| CoreML dynamic, ANE requested | 285 | 240 | 2.1×RT | falls back to **CPU** |
| ExecuTorch XNNPACK fp32 | — | 206 | 1.9×RT | CPU, 255 subgraphs |

**Why MLX beats XNNPACK by 4.6×:** the MLX partitioner reports *all ops supported* and produces
**one subgraph**. XNNPACK fragments the same graph into 255, stranding every RMSNorm, per-layer mask
conversion and shape-glue op on portable CPU kernels between delegate calls. ✅

**Why not CoreML,** despite the ANE being 1.7× faster at matched shapes: see §7.

---

## 5. ⚑ The governing constraint: 6.3 µs per graph node

Synthetic graphs of increasing depth, exported to MLX and timed: ✅

| nodes | ms |
|---|---|
| 2 | 0.201 |
| 200 | 1.489 |
| 840 | 5.498 |

**≈6.3 µs per node, plus a 0.2 ms per-call floor.** The step graph has 1948 nodes → ~12 ms of pure
interpreter cost, which matches the measured fixed term (step ≈ **9.5 ms fixed + 0.13 ms per target
frame** at P=166 📊).

The MLX delegate builds a lazy array graph and submits it via `async_eval` on a dedicated stream, so
this is CPU-side graph construction, not kernel launch latency.

**Everything else in this document follows from this one fact:**

- Quantisation cannot help — the bottleneck is not weight bandwidth (§6)
- fp32 is nearly free — precision does not add nodes (§6)
- Batching works — the interpreter cost is per *call*, not per item (§6)
- Prefix length barely matters — P from 60→380 moves the step only 22.3→25.3 ms
- The only latency lever is **node count**

---

## 6. What was tried

### Works ✅

| change | effect | accuracy |
|---|---|---|
| **fp32 `audio_heads`** | free (2 nodes) | **96.2% → 99.3% argmax, cb0 → 100%** |
| Native `F.rms_norm` | 1948 → 1183 nodes, **1.12×** | −0.6pt argmax |
| fp32 `q_norm`/`k_norm` | free | restores cb0 to 100% |
| Fused QKV projection | 1.03× | exact (weight concat + split) |
| Request batching (B=8) | **1.35× throughput** | exact |

**The fp32-logits finding is the important one.** Logits span ~127, where fp16 spacing is 0.0625,
and the measured fp16 error was ~0.09 — *at* that quantum. The dominant error was never the
arithmetic, it was **storing the output in fp16**, so near-ties flipped at random. Keeping only the
final 1024→8200 projection in fp32 removes rounding at the one place the sampler reads. It also
fixes the length sensitivity: baseline argmax decays 96.8% → 95.4% as T grows 75 → 350, while the
fp32 head holds flat near 99%. ✅

### Does not work ✅ (measured, negative)

| change | why not |
|---|---|
| **Weight quantisation** | see §8 — no speedup *and* severe accuracy loss |
| Fused gate/up projection | 0.99× — the 6144-wide split costs more than the launch saved |
| Stacking 42 K/V into one tensor | <1% — ExecuTorch passes by reference, nothing is copied |
| `mx.compile` | not reachable; the delegate ships its own FlatBuffer interpreter |

---

## 7. CoreML: fast on the ANE, but static-shape only

The ANE compiler **rejects any non-static graph**:

```
MILCompilerForANE error: failed to compile ANE model using ANEF.
Error=_ANECompiler : ANECCompile() FAILED.
```

Enumerated shapes do **not** avoid this — ranged and enumerated measured identically (236 vs 240 ms).
Worse, the fallback goes to **CPU, not GPU**, and `ComputeUnit.ALL` also lands on CPU; you must ask
for `CPU_AND_GPU` explicitly to get the GPU, which then matches MLX. ✅

**Static buckets do not pay off** on this corpus. Weighted over 264k utterances (median target
81 frames = 3.2 s, against a 125-frame bucket): 📊

| scheme | mean ms | vs MLX | win rate |
|---|---|---|---|
| MLX dynamic | 200.3 | 1.00× | — |
| CoreML, 2 target buckets (5 s / 10 s) | 187.6 | 1.07× | 51.0% |
| CoreML, 5 target buckets | 155.5 | 1.29× | 99.9% |

Two buckets is a wash. Five buckets is a real 1.29× but costs ~800 MB per graph (~5.6 GB) plus a
router and an overflow path — 14.6% of utterances exceed 250 frames. ⚠️ Sensitive to the length
distribution: this corpus is read-speech sentences, and traffic with longer utterances would move
CoreML back toward its 1.7×.

**Reference vs target granularity:** only the *target* bucket matters. Step cost tracks T almost
entirely, and the prefix runs once.

### Four conversion bugs, all needing workarounds ✅

1. **Bool mask in SDPA** — `mb.sub(x=1.0, y=mask)` mixes fp32 and fp16. Use a **zeros float mask**.
2. **fp16 attention scale** — SDPA decomposition multiplies fp16 by an fp32 scale. **Export the fp32
   graph** and set `compute_precision=FLOAT16`; CoreML converts internally. This is the recommended
   CoreML flow anyway.
3. **Enumerated shapes need `lower_full_graph=True`** in `CoreMLPartitioner` (asserted).
4. **Enumerated shapes accept only fp16/fp32/int32** — pass `input_ids` and `audio_mask` as **int32**
   and cast inside the graph.

CoreML also has no bf16: `ct.precision` is `FLOAT16` or `FLOAT32` only.

---

## 8. ⚑ This checkpoint is unusually fragile to weight quantisation

Int8 weight-only is normally near-lossless. Here it is not: ✅

| config | step ms | argmax | cb0 | size |
|---|---|---|---|---|
| fp16 | 23.8 | 96.20% | 98.40% | 1579 MB |
| int8 (`8w`) | 23.1 | **79.70%** | 76.80% | 923 MB |
| int4 (`4w`) | 22.4 | **12.30%** | 10.40% | 627 MB |

**This is not an MLX lowering bug.** The same quantised weights in eager PyTorch give 80.30% / 12.50%
— within a point of the delegate. ✅

Likely cause: `models/final` is already width-pruned, depth-pruned, vocab-pruned and distilled, so
there is little redundancy left to absorb quantisation noise. **Treat quantisation as unavailable on
this checkpoint** unless something changes upstream.

Note also that even int4 gave only 1.06× — further confirmation of §5. ⚠️ Untested: quantising the
backbone linears while leaving `audio_heads` in fp16, which might recover accuracy. It cannot help
speed, so it is only worth doing if file size becomes the constraint.

---

## 9. Precision menu

P=166, T=125. `fast` = fused QKV + native RMSNorm. ✅

| variant | nodes | step ms | vs base | argmax | cb0 |
|---|---|---|---|---|---|
| fp16 baseline | 1948 | 23.12 | 1.00× | 96.20% | 98.40% |
| fp16 + fp32 head | 1950 | 23.02 | 1.00× | 99.30% | 100.00% |
| **fast + fp32 head + fp32 qk** | 1395 | **20.92** | **1.11×** | **99.00%** | **100.00%** |
| fp32 whole model | 1948 | 28.30 | 0.82× | 100.00% | 100.00% |

**Recommended:** `fast + fp32 head + fp32 qk` — faster *and* more accurate than the fp16 baseline.

**Full fp32 costs only 22% on MLX** — a genuine option for exact parity, and in sharp contrast to
CoreML where fp32 cost 4.4×. Same reason: fp32 does not add nodes.

**bf16 is strictly worse than fp16** — 72.82% argmax at identical speed. It trades mantissa for
exponent (7 bits vs 10) and there was never any overflow to solve. Do not use it. ✅

---

## 10. Caveats

- **Argmax agreement is a proxy.** It is the right one — it is what the sampler consumes — but it
  cannot separate 99.00% from 99.30%. ⚠️ **Validate with WER before shipping any of these.**
- Backbone only. The generation loop (MaskGIT unmasking, step schedule) stays in Python, and the
  audio codec is a separate model that has not been exported.
- `models/final` is guidance-distilled, so there is **no CFG** — one forward per LM call, not two.
- Batching helps throughput, not latency, and saturates at ~1.35× by B=8.

---

## 11. Reproducing

Scripts are not in the repo (kept in scratch during exploration). The essentials:

```python
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

ep = export(step_module, step_inputs, dynamic_shapes=ds, strict=False)   # strict=False for HF
prog = to_edge_transform_and_lower(ep, partitioner=[MLXPartitioner()]).to_executorch()
```

Gotchas: `torch.export` **specialises a size-1 dim**, so trace a batch dim at 2 if it must stay
dynamic; varargs (`*past`) are **one** element in the `dynamic_shapes` tree, not 42; and MLX requires
**contiguous** inputs — call `.contiguous()` on every slice.

Set `ET_MLX_DEBUG=1` during export for partitioner and per-node decisions.
