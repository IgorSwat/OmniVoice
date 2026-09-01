# OmniVoice → Student: Distillation & Compression Plan

Analysis of `k2-fsa/OmniVoice` for the purpose of distilling a smaller, faster,
language-restricted student model.

**Goal:** 612M → ~200M params, 600+ languages → 10, 32 diffusion steps → 16,
targeting on-device voice-clone-only inference.

**Provenance markers used throughout:**

| Marker | Meaning |
|---|---|
| ✅ | **Measured** — I ran this against the local checkpoint / files |
| 📊 | **Computed** — arithmetic from verified config values |
| ⚠️ | **Unverified** — from memory or a single source; confirm before relying on it |

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Architecture analysis](#2-architecture-analysis)
3. [The audio codec](#3-the-audio-codec)
4. [Target student configuration](#4-target-student-configuration)
5. [The optimization stack](#5-the-optimization-stack)
6. [Compression method (Phases A–E)](#6-compression-method-phases-ae)
7. [Compute budget & hardware](#7-compute-budget--hardware)
8. [Data requirements](#8-data-requirements)
9. [Dataset map](#9-dataset-map)
10. [Filtering pipeline](#10-filtering-pipeline)
11. [Measured reference values](#11-measured-reference-values)
12. [Constraints and dead ends](#12-constraints-and-dead-ends)
13. [Recommended order of work](#13-recommended-order-of-work)

---

## 1. Executive summary

The compression is viable, and the architecture is unusually friendly to it.

**Results of the analysis:**

- **~21× faster** end-to-end and **~3.6× smaller**, compounding five independent levers
- On-device footprint drops to **~0.36 GB** (fp16 LM + codec decoder), from a ~1.5 GB teacher stack
- **Compute is not the bottleneck** — the full pipeline is 1–3 days on 4 GPUs. Engineering time is
- The two highest-value experiments require **zero training data** and can be run today

**Three structural properties make this work:**

1. The 155M input embedding is **untied from any output head** (`audio_heads` is the only
   output), so vocab pruning is exactly lossless for unused tokens
2. Qwen3 is **pre-norm RMSNorm with no biases**, satisfying SliceGPT's preconditions
   without any conversion step
3. `head_dim: 128` is **explicit and decoupled** from `hidden_size`, so the residual stream can
   shrink without touching RoPE or `q_norm`/`k_norm`

**The one hard constraint:** the codec cannot be modified. Every teacher token and cached
trajectory is defined by its exact RVQ codebooks.

---

## 2. Architecture analysis

`OmniVoice` ([`omnivoice/models/omnivoice.py:200`](OmniVoice/omnivoice/models/omnivoice.py#L200))
is a thin wrapper around a **Qwen3-0.6B backbone used as a bidirectional masked-diffusion LM**.

### Components

- `self.llm` = `AutoModel.from_config(qwen3_config)` → a plain `Qwen3Model`
  (28 layers, hidden 1024, head_dim 128, 16 Q / 8 KV heads, intermediate 3072, **no `lm_head`**)
- `audio_embeddings`: `nn.Embedding(8×1025, 1024)`, **summed** over the 8 codebooks
- `audio_heads`: `nn.Linear(1024, 8×1025, bias=False)` — the only output head
- Attention is **fully bidirectional** — `_mask_mod_packed` is document-membership only,
  and [`collator.py:41-43`](OmniVoice/omnivoice/data/collator.py#L41-L43) confirms no causal mask

### Parameter budget ✅

Measured from `model.safetensors` (313 tensors, 612.58M params):

| Block | Params | Share |
|---|---|---|
| `llm.embed_tokens` (151676 × 1024) | 155.3M | 25.4% |
| 28 × attention (q/k/v/o) | 176.2M | 28.8% |
| 28 × MLP (gate/up/down) | 264.2M | 43.1% |
| `audio_embeddings` + `audio_heads` | 16.8M | 2.7% |
| norms | 0.06M | — |
| **Total** | **612.6M** | |

Per layer: 15.73M (attn 6.30M, mlp 9.44M).

### Generation loop

`_generate_iterative` ([`omnivoice.py:1153`](OmniVoice/omnivoice/models/omnivoice.py#L1153)) is
MaskGIT-style confidence-based iterative unmasking:

- **No KV cache.** Every step is a full bidirectional forward over the whole sequence
- **CFG doubles the batch** — cond over `S`, uncond over the target region only
- Cost per generation = `32 steps × 2 branches` = up to 64 forwards
- Position selection: top-k on `log_probs.max()` with gumbel noise (`position_temperature=5.0`)
  and a layer penalty (`layer_penalty_factor=5.0`) biasing early codebooks to resolve first
- Each slot is committed **exactly once** — `scores.masked_fill_(sample_tokens != mask_id, -inf)`

### Voice cloning is self-supervised ✅

Critically: the clone prompt is a **prefix of the same utterance**
([`processor.py:132`](OmniVoice/omnivoice/data/processor.py#L132)),
with `prompt_ratio ~ U(0.0, 0.3)` and loss suppressed on the prompt region.

**Consequences:**

- Your corpus needs **no paired reference audio** — single utterances with transcripts suffice
- There is **no cross-utterance speaker mechanism** anywhere: no speaker embedding table, no
  speaker ID token. Per-speaker hours is a *sampling* knob, not a modeling parameter
- Utterance duration matters: a ≤30% prefix of a 10s clip is ≤3s of reference

### Auto mode is not separable from CFG ✅

When `drop_cond` fires, the processor sets `drop_text=True` and the input becomes **audio tokens
only**, with no style/text prefix
([`processor.py:150-154`](OmniVoice/omnivoice/data/processor.py#L150-L154)).
Inference does the same for the unconditional branch:
`batch_input_ids[B+i] = inp["input_ids"][..., -u_len:]`
([`omnivoice.py:1218`](OmniVoice/omnivoice/models/omnivoice.py#L1218)).

**They are the same code path.** You can only drop auto-mode capacity if you do guidance
distillation. This promotes guidance distillation from "nice speedup" to "the mechanism that
lets you drop a mode."

### Language conditioning

Injected as **plain text** — `<|lang_start|>en<|lang_end|>`
([`processor.py:106`](OmniVoice/omnivoice/data/processor.py#L106)). 646 entries in `LANG_IDS`,
**zero per-language parameters.** All multilingual capacity is diffuse in the FFN weights.

This is why calibrating on 10 languages works: neurons that fire only for the other 636 score
near zero in activation-based selection and get dropped automatically. It's the mechanism that
makes 3× compression plausible rather than wishful.

---

## 3. The audio codec

**Higgs Audio V2 Tokenizer** (`HiggsAudioV2TokenizerModel`), bundled in the checkpoint at
`audio_tokenizer/`. Loaded at
[`omnivoice.py:277-279`](OmniVoice/omnivoice/models/omnivoice.py#L277-L279).

### Properties ✅

| Property | Value |
|---|---|
| Input / output sample rate | **24,000 Hz** |
| Token frame rate | **25 Hz** (hop 960 = 8×5×4×2×3) |
| Internal semantic branch | 16,000 Hz (HuBERT operates there) |
| Quantizer | 8× residual VQ, 1024 entries × 64 dim |
| Bitrate | **2 kbps** (8 × 10 bits × 25 Hz) |
| Parameters (file) | 201.40M, 527 tensors |
| File size | 805.7 MB / 768.3 MiB, fp32 |
| Encode path | 160.4M params |
| **Decode path** | **21.56M params** |

### Module breakdown ✅

| Module | Params | Path |
|---|---|---|
| `semantic_model` — HuBERT-base (12 layers, d=768) | 94.37M | encode |
| `acoustic_encoder` — DAC-style conv | 51.31M | encode |
| `encoder_semantic` | 14.75M | encode |
| `acoustic_decoder` | 20.24M | **decode** |
| `decoder_semantic` | 16.52M | *unused at inference* |
| `fc` / `fc1` / `fc2` | 2.10M | only `fc2` used |
| `quantizer` (8× RVQ) | 1.06M | shared |

**Verified by forward hooks:** `decode()` executes only `acoustic_decoder`, `quantizer`, and
`fc2`. `decoder_semantic`, `fc`, and `fc1` are training-time auxiliary heads (`fc1` maps to 768 =
HuBERT hidden size, i.e. a semantic reconstruction loss).

### Shippable decoder ✅

| | Value |
|---|---|
| Parameters + buffers | **21.56M** (136 tensors) |
| fp32 | 82.2 MiB |
| **fp16** | **41.1 MiB** |
| int8 | 20.6 MiB |
| vs. full codec file | 768.3 MiB (**11%**) |

**Functionally verified:** replacing all six unused submodules with `nn.Identity()` and dropping
`project_in` plus the EMA buffers leaves a 21.56M model whose decode output is **bit-identical**
to the full model's.

### Speed ✅

Apple Silicon, 8 CPU threads, fp32 — no GPU, no quantization:

| | Latency | RTF | ×realtime |
|---|---|---|---|
| decode 5s | 101 ms | 0.0203 | 49× |
| decode 10s | 181 ms | 0.0181 | **55×** |
| decode 30s | 558 ms | 0.0186 | 54× |
| encode 10s | 275 ms | 0.0275 | 36× |
| encode 30s | 868 ms | 0.0289 | 35× |

Linear in duration — the decode path is fully convolutional, no attention.

### Encoder is needed once per reference ✅

`self.audio_tokenizer.encode(...)` appears at **exactly one call site** in the whole model:
[`omnivoice.py:701`](OmniVoice/omnivoice/models/omnivoice.py#L701), inside
`create_voice_clone_prompt()`. And `generate()` short-circuits it — `create_voice_clone_prompt` is
called only when `voice_clone_prompt is None and ref_audio is not None`
([`omnivoice.py:952`](OmniVoice/omnivoice/models/omnivoice.py#L952)).

```python
# offline, once per speaker
prompt = model.create_voice_clone_prompt(
    ref_audio="data/morgan/reference.wav",
    ref_text=open("data/morgan/transcript.txt").read().strip(),
)
torch.save(prompt, "data/morgan/prompt.pt")   # ~2-4 KB

# on-device, every generation — no encoder involved
audio = model.generate(text="...", voice_clone_prompt=torch.load("data/morgan/prompt.pt"))
```

A cached "voice" is **2–4 KB** (`ref_audio_tokens` int16 + transcript + one float).

**What this removes from the device:**

- codec encoder (160M, ~320 MB fp16)
- Whisper `large-v3-turbo` (~800M) — only used when `ref_text is None`

**Caveats:**
- A prompt is per *recording*, not per speaker in the abstract — different styles need different prompts
- `ref_text` is bound to the recording. The code skips `trim_long_audio` when `ref_text` is
  user-supplied ([`omnivoice.py:672-676`](OmniVoice/omnivoice/models/omnivoice.py#L672-L676)) to
  avoid desynchronization, so **trim references to 3–10s yourself**
- The stripped deployment still needs the tokenizer *object* (config reads at lines 581, 696), just not its encoder weights

---

## 4. Target student configuration

```python
hidden_size          = 704      # from 1024
num_hidden_layers    = 20       # from 28
intermediate_size    = 1536     # from 3072
num_attention_heads  = 12       # from 16  (group prune 8→6)
num_key_value_heads  = 6        # from 8
head_dim             = 128      # unchanged — keeps RoPE and q_norm/k_norm intact
vocab_size           = ~40000   # from 151676
```

### Size 📊

| | embed | attn | mlp | audio | **Total** | vs teacher |
|---|---|---|---|---|---|---|
| Teacher (d1024, L28, I3072) | 155.3M | 176.2M | 264.2M | 16.8M | **612.6M** | 1.00× |
| d704, L20, I3072, groups 8→8 | 106.8M | 86.5M | 129.8M | 11.5M | **334.6M** | 0.55× |
| + vocab 152k→40k | 28.2M | 86.5M | 129.8M | 11.5M | **256.0M** | 0.42× |
| **+ I 3072→1536, groups 8→6** | 28.2M | 64.9M | 64.9M | 11.5M | **169.5M** | **0.28×** |

**Why d704/L20 alone is only 45%:** transformer blocks scale linearly in both `d` and `L`
(440.5M × 704/1024 × 20/28 = 216.3M ✓), but the **embedding doesn't shrink with depth at all**.
It survives as 32% of the un-pruned student — larger than either the attention or MLP totals.
Vocab pruning is worth 78.6M, more than the depth cut.

### Depth vs width

Prefer **depth** at equal budget. You're halving diffusion steps, and fewer steps means each
forward must do more work per call — depth is what buys per-forward compute. Cutting depth and
steps simultaneously attacks the same resource twice.

Alternates at the same budget: `L24/d640/I1536` (deeper), `L18/d768/I1792` (wider).

Note: with d=704 and head_dim=128 × 12 heads, attention projections are 1536-wide against a
704-wide residual. Legal in `Qwen3Config` (head_dim is explicit) and it's what keeps RoPE
untouched.

---

## 5. The optimization stack

End-to-end FLOPs for one 10s generation with a 5s reference (P=200 prefix, T=250 target, S=450). 📊

| Stage | GFLOP | Cumulative | This step |
|---|---|---|---|
| Teacher: 32 steps + CFG, full attention | 28,340 | 1.00× | — |
| **+ model size** (d704 / L20 / I1536 / groups 8→6) | 9,067 | 3.13× | 3.13× |
| **+ 16 steps** (`t_shift=0.2`) | 4,533 | 6.25× | 2.00× |
| **+ CFG eliminated** (guidance distillation) | 2,267 | 12.50× | **2.00×** ✅ |
| **+ prefix K/V blocking** | 1,316 | **21.53×** | 1.72× |
| + hybrid sliding window (4 of 20 full, W=256) | 1,240 | 22.86× | 1.06× |

**CFG costs a true ~2×.** An earlier version of this document estimated 1.51×, reasoning that the
unconditional branch runs over only the target region. That is wrong: `_generate_iterative` builds
`batch_input_ids` as `(2B, C, max_c_len)` and **pads the unconditional half to the same length**
([`omnivoice.py:1195-1224`](OmniVoice/omnivoice/models/omnivoice.py#L1195-L1224)), with pad
positions masked to attend only to themselves. The compute still happens over the full length.
Measured forward-time ratio B=2 vs B=1: **1.91–2.04×** across S=256–1024. ✅

### 5.1 Fewer diffusion steps — and the `t_shift` finding ✅

The unmasking schedule is heavily **back-loaded**. For a 10s utterance (2000 masked slots at
25 Hz × 8 codebooks), as fractions of total:

```
n=32, t_shift=0.1:  per-step [0.0032, 0.0034, ...] ... last step 0.244
n=16, t_shift=0.1:  per-step [0.0066, 0.0075, ...] ... last step 0.400
```

The failure mode of parallel unmasking is a **conditional-independence violation**: committing
`k` tokens in one step samples from the product of marginals, not the joint. The 0.244 → 0.400
jump is where quality goes — not general model inaccuracy.

Sweeping `t_shift` at n=16:

| `t_shift` | max per-step | first step |
|---|---|---|
| 0.1 | 0.400 | 0.0066 |
| **0.2** | **0.250** | 0.0132 |
| 0.3 | 0.182 | 0.0196 |
| 0.5 | 0.118 | 0.0323 |

**`t_shift=0.2` at 16 steps reproduces the 32-step maximum parallel-commit fraction almost
exactly** (0.250 vs 0.244). One config field, zero training. **Run this on the teacher first** —
it tells you how much of the 32→16 gap is schedule shape versus genuine model capacity.

(The back-loading is deliberate and fine at 32 steps: `layer_penalty_factor=5.0` forces codebooks
0/1 to resolve during the early fine-grained steps, so the terminal dump is mostly high codebooks
that are cheap and highly predictable given the lower ones.)

### 5.2 Prefix K/V blocking 📊

The sequence is `[style | text | ref_audio | target_audio]`. During the step loop, **only the
target region changes** — but because attention is fully bidirectional, the prefix's hidden states
depend on the target, so all of it is recomputed 16–32 times.

**Mask prefix→target attention** (prefix attends only to itself; target still attends to
everything). The prefix forward runs once, its K/V cache per layer, and each step processes only
the target positions:

| P | T | Full/step | Blocked/step | Speedup |
|---|---|---|---|---|
| 150 | 125 | 4.78G | 2.17G | **2.20×** |
| 200 | 250 | 8.47G | 4.71G | **1.80×** |
| 250 | 500 | 15.96G | 10.64G | 1.50× |
| 300 | 750 | 24.93G | 17.81G | 1.40× |

End-to-end for P=200, T=250, 16 steps including the one-time prefix pass: **1.72×**.

This isn't only an attention saving — you skip the prefix's **entire forward pass** (projections
and MLP) on 15 of 16 steps.

**Why it's sound:** the target still sees the full prefix; only the reverse direction is cut. The
loss is computed only on the target region anyway — `audio_labels[:, :prompt_length] = -100`,
with text and style labels also `-100`
([`processor.py:145-152`](OmniVoice/omnivoice/data/processor.py#L145-L152)). This is the
prefix-LM / encoder-decoder split.

**Why it's cheap:** it's a **mask change, not architecture surgery**. The model already accepts a
4D attention mask, and [`collator.py:90`](OmniVoice/omnivoice/data/collator.py#L90) already builds
one position-by-position. Add a `mask_mod` variant alongside `_mask_mod_packed` for
flex_attention, and modify inference mask construction at
[`omnivoice.py:1206-1224`](OmniVoice/omnivoice/models/omnivoice.py#L1206-L1224).

**Validate with zero training:** apply the blocked mask to the *teacher* and measure
WER/SIM/UTMOS.

### 5.3 Attention cost vs sequence length 📊

Student (d=704, I=1536, L=20): per layer per token, proj+mlp = 15.14 MFLOP, attention = 8.2k × S.

| Scenario | S | Attention share |
|---|---|---|
| 5s ref + 5s target | 450 | 19.6% |
| 5s ref + 10s target | 575 | 23.7% |
| 5s ref + 20s target | 875 | 32.1% |
| 5s ref + 30s target *(chunk cap)* | 1175 | 38.9% |
| 60s, chunking off | 2025 | 52.3% |

Attention never runs away because **chunking already bounds S**: `audio_chunk_threshold: 30.0`
splits longer text and generates chunk-by-chunk with cross-fading.

### 5.4 KV groups vs KV merging 📊

Qwen3-0.6B uses GQA: 16 query heads share 8 KV heads, so each KV head serves **2** query heads.
That pairing is a **group**. GQA requires `H % Kv == 0` with equal group sizes, so the unit of
removal is the group.

At d=704, L=20:

| | H | Kv | ratio | per layer | ×20 | params saved | **S² attn cost** |
|---|---|---|---|---|---|---|---|
| baseline | 16 | 8 | 2× | 4.325M | 86.5M | — | 100% |
| **merge** KV 8→4 | 16 | 4 | 4× | 3.604M | 72.1M | 16.7% | **100%** |
| **merge** KV 8→2 | 16 | 2 | 8× | 3.244M | 64.9M | 25.0% | **100%** |
| **prune** groups 8→6 | 12 | 6 | 2× | 3.244M | 64.9M | 25.0% | **75%** |
| **prune** groups 8→4 | 8 | 4 | 2× | 2.163M | 43.3M | 50.0% | **50%** |

**Attention's S² cost scales with `H`, not `Kv`** — the score matmul expands K from
`[Kv, S, hd]` to `[H, S, hd]` before multiplying, so cost is `H × S × S × hd` regardless.
Merging KV heads does **nothing** for the sequence-length problem.

And GQA's usual motivation — shrinking the KV cache during autoregressive decoding — **doesn't
apply here at all**, because there is no KV cache. Every step recomputes everything.

**Therefore: prune whole groups (8→6), don't merge.** Same parameter saving, plus 25% off the
S² term.

**Surgery:** delete from every layer the 2×128 rows of `q_proj`, the 128 rows of `k_proj`/`v_proj`,
and the corresponding 256 columns of `o_proj`. `q_norm`/`k_norm` are `[128]` — per-`head_dim`,
shared across heads — so they're **untouched**, as is RoPE.

**Selection:** contribution norm (‖o_proj slice @ head output‖ over calibration data) or
leave-one-out loss. Prune per layer; if config uniformity forces a global count, rank by *worst*
per-layer score. Then least-squares repair `o_proj` on survivors.

### 5.5 Sliding-window attention

Natively supported: the teacher config carries `layer_types: ["full_attention" × 28]`,
`sliding_window: null`, `use_sliding_window: false`, `max_window_layers: 28`. HF's Qwen3 honors a
**per-layer** `layer_types`, so a hybrid (one full layer every 4–6) is expressible in config
with no code changes.

Speech has strong temporal locality, but text conditioning and the reference prompt are *global*
— so the natural pattern is **global attention to the prefix, local within the target**, which
composes cleanly with prefix blocking.

**Decide by measurement in the Phase B pass:** add attention-weight capture and compute, per
layer, what fraction of attention mass falls within ±W frames. Layers concentrating >90% locally
convert essentially free.

Only worth the risk at long outputs — it adds just 1.06× at S=450.

### 5.6 Free levers, no training

- **Vocab pruning** — 78.6M params, exactly lossless for unused tokens
- **`t_shift=0.2`** — the enabler for 16 steps
- **Codec decoder-only deployment** — 201M → 21.6M, verified bit-identical ✅
- **Precomputed voice prompts** — drops 160M encoder + ~800M Whisper from the device
- **Shorter reference audio** — P scales with it; 3s vs 8s is real compute
- **Lower `audio_chunk_threshold`** (30s → 15s) — ~4× off attention in the worst case, at the
  cost of more cross-fade seams

### 5.7 Implementation — possibly the largest real-world effect

- **The per-item sampler loop is NOT a bottleneck** — measured. An earlier version of this
  document called vectorizing [`omnivoice.py:1268`](OmniVoice/omnivoice/models/omnivoice.py#L1268)
  "probably the highest-ROI hour in the project." That was wrong. Measured on M4 Pro / MPS it is
  **0.1% of total time at B=1, 4, and 8** (§11.1). It scales linearly in B and stays negligible.
  It could still matter on hardware where a forward takes single-digit ms — at B=8 it costs
  ~1.1 ms/step, which would be ~10% against a 10 ms forward — but it is nowhere near the
  first-order concern claimed here originally. Revisit only after profiling on the target GPU.
- **Raise `batch_tokens`** from the 8192 default. At that size a student step is ~43 ms — too
  short to hide kernel launch, PCIe all-reduce, and data loading. Expect ~10% MFU instead of 30%.
  Use 65536 (32 GB cards) or 131072 (96 GB). Scale `steps` down proportionally and raise LR.

### 5.8 Codebook count — indirect

The codec supports 0.5/1/1.5/2 kbps → 2/4/6/8 codebooks via `encode(..., bandwidth=)`.
OmniVoice hardcodes 8.

Round-trip mel-L1 on LibriTTS ✅:

| | 2.0 kbps / 8cb | 1.5 / 6cb | 1.0 / 4cb | 0.5 / 2cb |
|---|---|---|---|---|
| mel-L1 vs original | 0.638 | 0.698 (+9%) | 0.774 (+21%) | 1.03 (+60%) |

Dropping to **6 codebooks costs ~9%** and cuts masked slots 25% (2000 → 1500 for 10s), which
directly eases the 16-step conditional-independence problem. Codebooks 6–7 carry only 4/40 of the
loss weight. Truncate the teacher's tokens to the first 6 and train the student on 6 — safe
because it's a prefix truncation of the same RVQ.

**Does NOT shorten the sequence.** Codebooks are summed into one embedding per frame
([`omnivoice.py:373`](OmniVoice/omnivoice/models/omnivoice.py#L373)), so S is *frames*,
independent of C. Validate with UTMOS before committing.

---

## 6. Compression method (Phases A–E)

### Phase A — vocab pruning

The input embedding is a **pure lookup with no output-side constraint** (`tie_word_embeddings:
true` is in `llm_config`, but `AutoModel` instantiates `Qwen3Model`, which has no `lm_head`; the
checkpoint confirms it). Rows for tokens your tokenizer never emits are exactly dead weight.

**Use your training transcripts as the frequency source** — they *are* the training distribution,
and they're spoken-style by construction. Caveats:

- **Volume:** ~150 words/min → 200h ≈ 2.4M tokens, 1,000h ≈ 12M, 3,000h ≈ 36M. Thin at
  experiment scale
- **ASR transcripts are circularly biased** — ASR under-produces rare words, and Whisper writes
  digits ("23") while OmniVoice's README tells users to normalize to words ("twenty-three")
- **Train ≠ inference** — users type arbitrary text. Byte-level BPE means never OOV, just more splits

**The cost curve makes this low-stakes.** At d=704, one vocab row = **704 params**:

- 151,676 → 40,000 saves **78.6M** — the big win
- 40,000 → 30,000 saves only **7M**
- 20,000 extra tokens costs 14M ≈ 6.8% of a 206M student

So **set the cutoff generously** (50–60k), union the transcripts with a modest general corpus, and
keep all 256 byte tokens + the 7 added specials. Count **post-filtering** and **with the
processor's actual formatting** (style prefix, `<|text_start|>` wrapper, `add_punctuation`).

**Defer Phase A entirely for early experiments** — it changes size, not quality, and is orthogonal
to every question the early experiments answer.

### Phase B — structural surgery

Instrument the calibration set, then:

1. **Width — residual-stream PCA (1024 → 704).** Pool residual snapshots into one covariance,
   eigendecompose, take the top-`d'` eigenvectors as `Q`, then transform:
   - `E_text ← E_text Q`, `audio_embeddings ← audio_embeddings Q`
   - `q/k/v_proj ← Qᵀ W`, `o_proj ← W Q`
   - `gate/up_proj ← Qᵀ W`, `down_proj ← W Q`
   - `audio_heads ← audio_heads Q`
   - rescale by `sqrt(d'/d)` folded into the following weights (RMS is a mean over dims)

   **Use one global `Q` shared across all layers, not per-layer.** SliceGPT uses per-layer bases
   and must insert adapter matrices at every residual connection — that changes the architecture.
   A shared basis is well-motivated for a pre-norm net (the residual stream is one accumulating
   buffer) and keeps the student a stock Qwen3.

   Fold each RMSNorm's learnable gain `g` into the following linear layer first, leaving pure RMS
   normalization, which commutes with any orthogonal `Q`.

2. **Depth (28 → 20).** Block-influence: per layer, `1 − cos(x_in, x_out)` averaged over
   calibration tokens. Drop the lowest-influence span. ⚠️ That redundancy result was established
   on *causal* LMs; this model is bidirectional, so **re-measure rather than assume**.

3. **FFN neurons (3072 → 1536).** You cannot rotate the FFN inner space — SiLU sits between
   `gate/up` and `down`, so only axis-aligned **selection** is valid. Score by
   `mean|activation| × ‖down_proj row‖`.

4. **KV groups (8 → 6)** — see §5.4.

5. **Least-squares repair.** After every selection step, refit the surviving output projection
   against the teacher's output on cached activations:
   `min_W ‖W_teacher·h_teacher − W·h_student‖²`, ridge-regularized. Closed-form, minutes.
   **Weight by `normalized_audio_codebook_weights`** — `[8,8,6,6,4,4,2,2]` means codebook 0 carries
   4× the weight of codebook 7.

**Memory note:** everything is accumulable in second-moment form. Never store raw activations —
accumulate covariance (1024², 4 MB/boundary) and `XᵀX`/`XᵀY` for repair. Memory is O(d²), not O(N·d).

**Gate:** measure the student at 32 steps with CFG *before* touching the sampler. This is the
single most informative number in the project and it costs nothing.

### Phase C — recover at 32 steps

KD on the target languages:
- Soft KL over the 1025-way distribution per codebook, weighted by `audio_codebook_weights`
- Hidden-state matching against `h_teacher · Q` — **the PCA basis gives you the alignment map for
  free**, no learned projection head
- Mixed with some fraction of the original ground-truth CE so the student doesn't overfit teacher quirks

[`trainer.py:284-285`](OmniVoice/omnivoice/training/trainer.py#L284-L285) is
`outputs = self.model(**batch); loss = outputs.loss` — a single clean hook to subclass.

**Keep the teacher resident.** Precomputing KD targets saves ~0.75 h/epoch and breaks even after
~4 epochs, but costs 824 GB of cache management, freezes your mask realizations, and needs
~570 MB/s sustained read. At this scale that trade isn't worth it.

### Phase D — step + guidance distillation

Progressive distillation adapted to masked diffusion: run the teacher's 32-step loop with CFG,
cache the trajectory, then train the student to go from `x_2i` to `x_{2i+2}` in one step — CE on
the tokens the teacher committed across those two steps.

**Convenient property: the model has no timestep embedding.** `forward()` takes only
`input_ids`/`audio_mask`/masks, so the noise level is inferred from the mask pattern and there's
nothing to re-condition.

**Trap: match the teacher's confidence distribution, not just its argmax.** The sampler uses
`log_probs.max()` to choose *which* positions to commit
([`omnivoice.py:1327`](OmniVoice/omnivoice/models/omnivoice.py#L1327)), with gumbel temperature
5.0. A student that gets every argmax right but has miscalibrated confidence will unmask in the
wrong order and degrade badly. Full soft KD handles this; argmax-only KD does not.

**Guidance distillation** folds in here at no extra cost — train the student's single conditional
pass to reproduce the CFG-mixed teacher output, then set `guidance_scale=0` and edit the batch
doubling at [`omnivoice.py:1212-1224`](OmniVoice/omnivoice/models/omnivoice.py#L1212-L1224).

**Caching:** cache the **trajectories** (committed tokens + commit-step index, ~5 KB/utterance,
~6 GB corpus-wide) since generating them is the expensive part. Soft targets are recomputed by the
resident teacher.

Storage arithmetic if you did want to cache targets (3,000h ≈ 1.08M utterances) 📊:

| Artifact | Size |
|---|---|
| Full fp16 logits, all slots, one forward | 4.1 TB |
| top-64 (value+index, 4B/entry), one forward | 515 GB |
| Phase C targets, top-64, 4 mask realizations | 824 GB |
| Phase D off-policy targets, top-64 | 515 GB |
| **Trajectory tokens + commit-step index** | **6 GB** |
| Hidden states, all 22 boundaries | 12.7 TB |
| Hidden states, 4 boundaries @ 25% positions | 577 GB |

Each slot is committed **exactly once** across the trajectory
([`omnivoice.py:1291-1293`](OmniVoice/omnivoice/models/omnivoice.py#L1291-L1293)), so a full
16-step trajectory needs targets for 2000 slots total — same as a single forward, not 16×.

**Off-policy vs on-policy:** off-policy (teacher states) is fully precomputable and is what
progressive distillation does. On-policy (teacher's opinion about the *student's* states) is
genuinely uncacheable. Do off-policy first; reserve a short on-policy pass for the end, only if
eval shows trajectory drift.

**Validate top-k per codebook** before committing to a cache format: codebook 0 is likely peaked,
codebooks 6–7 are residual/acoustic and may have heavier tails. Measure cumulative mass at
k=32/64/128 *per codebook*.

### Phase E — re-tune the sampler

`t_shift`, `layer_penalty_factor`, `position_temperature`, `class_temperature` were tuned for the
teacher at 32 steps with CFG. Re-sweep all four on the final student.

---

## 7. Compute budget & hardware

**Compute is not the bottleneck.** These are small models on a small corpus — 3,000h ≈ only
~300M token-positions.

### Per-phase, 3,000h corpus 📊

| Phase | FLOPs | 4× RTX 6000 Ada | 4× RTX PRO 6000 |
|---|---|---|---|
| A — audio tokenization (one-time) | — | ~4 h | ~2.5 h |
| B — profiling + surgery | 0.004 EF | **minutes** | minutes |
| C0 — teacher-resident warmup, 10k steps | 1.0 EF | 1.3 h | 0.8 h |
| C — KD training, **per epoch** | 0.9 EF | 1.2 h | 0.73 h |
| C — 20 epochs | 18 EF | 23.6 h | **14.6 h** |
| D1 — trajectories, full corpus | 17.3 EF | 22 h | 13.7 h |
| D1 — trajectories, 15% subset | 2.6 EF | 3.3 h | 2.1 h |
| D2 — step-distill training | 0.6 EF | <1 h | <1 h |

**Full pipeline: ~24–60 GPU-hours across 4 cards — one to three days.**

At 10,000h everything scales ~3.3×: still days, not weeks. For scale, the teacher's own 2M-step
multilingual run is ~43 EF ≈ 2.5 days on the same 4 GPUs — your entire distillation project costs
about as much as retraining the teacher once.

Sanity check: the README quotes RTF 0.025 (40× realtime). 3,000h ÷ 40 ÷ 4 GPUs ≈ 19h, against the
22h D1 estimate. The FLOP model is sound.

### Memory

Teacher-resident Phase C, per GPU 📊:

| `batch_tokens` | Student act. | Teacher hidden | Base | Total | 32 GB? | 96 GB? | Step |
|---|---|---|---|---|---|---|---|
| 16,384 | 3.8 | 0.7 | 3.8 | 10.3 GB | ✅ | ✅ | ~0.25 s |
| 32,768 | 7.6 | 1.4 | 3.8 | 14.8 GB | ✅ | ✅ | ~0.50 s |
| **65,536** | 15.1 | 2.8 | 3.8 | **23.7 GB** | ✅ | ✅ | ~1.0 s |
| 98,304 | 22.7 | 4.1 | 3.8 | 32.6 GB | ❌ | ✅ | ~1.5 s |
| **131,072** | 30.2 | 5.5 | 3.8 | **41.6 GB** | ❌ | ✅ | ~1.7 s |

**Gradient checkpointing is not wired into this repo** (no references anywhere in `omnivoice/`).
`Qwen3Model` supports it natively — one line `model.llm.gradient_checkpointing_enable()` in the
builder cuts activations ~5–8× for ~30% extra compute.

### Hardware notes

- **No NVLink** on RTX 6000 Ada, RTX PRO 6000 Blackwell, or 5090. DDP all-reduce over PCIe:
  ~400 MB bf16 gradients, ~40–60 ms. At `batch_tokens=8192` that's >100% overhead; at 131072 it's
  under 10%. Another reason to go large-batch.
- **RTX 5090 (32 GB):** viable. Same 1.79 TB/s bandwidth as the PRO 6000, ~85–90% of its BF16
  throughput. Use `batch_tokens=65536`. ⚠️ Unknown: whether Blackwell consumer retains the
  half-rate FP32-accumulate penalty that the 4090 had — if it does, everything roughly doubles.
  Even then the full pipeline is a long weekend. **Measure one epoch and rescale.**
- **Cooling is a harder problem than VRAM** for 4× consumer cards — 4 × 575 W triple-slot
  blower-less cards in one chassis will throttle. FE cards, open frame, or watercooling.
- **Power:** 2.3–2.4 kW GPU, ~3–3.5 kW system. Needs 240 V or split circuits.
- **No ECC** on consumer cards — checkpoint frequently.

---

## 8. Data requirements

### Voice-clone-only scope

| Field | Value | Why |
|---|---|---|
| `instruct_ratio` | `0.0` | no voice design |
| `only_instruct_ratio` | `0.0` | no design-only path |
| `language_ratio` | `1.0` | few languages, always known |
| `use_pinyin_ratio` | `0.3` if zh else `0.0` | pronunciation control |
| `prompt_ratio_range` | `[0.1, 0.4]` | specialize to inference-length references |
| `drop_cond_ratio` | `0.1` in Phase C → `0.0` after guidance distillation | this is the auto/CFG path |

**Trap:** `<|instruct_start|>None<|instruct_end|>` is emitted **unconditionally** at
[`processor.py:105`](OmniVoice/omnivoice/data/processor.py#L105), even with `instruct_ratio=0`.
Leave it in through Phases B and C — stripping it changes sequence topology and breaks
teacher↔student activation alignment for hidden-state matching.

### Data format

```jsonl
{"id": "s001", "audio_path": "/abs/path/001.wav", "text": "Hello world", "language_id": "en"}
```

```bash
python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl data.jsonl \
    --tar_output_pattern out/audios/shard-%06d.tar \
    --jsonl_output_pattern out/txts/shard-%06d.jsonl \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer
```

### Sizing

| Stage | Audio needed |
|---|---|
| Sampler sweeps (`t_shift`, prefix blocking, codebook count) | **zero** — test sentences |
| Phase A (vocab) | text only, ~10M tokens |
| **Phase B (profiling + surgery)** | **2–6 h** (500–2,000 utterances) |
| **Init-quality eval** | **1 h held-out speakers** |
| **Recovery experiment** | **50–200 h** |
| Credible single-language student | 500–1,000 h |
| Full multilingual student | 1,000 h/language, 300 h floor |

**Why 2–6 h suffices for Phase B:** the residual covariance is 1024×1024 and you want ~10× the
dimension in *effective* samples. Adjacent frames are highly correlated, so budget ~10–30 effective
samples per utterance, not 280. At 1,000 utterances (~2.8 h) that's ~20k effective samples. The
tighter constraint is the FFN repair, whose `XᵀX` is 3072² — lean toward 2,000 utterances and
ridge-regularize. Sample each utterance at ~8 mask ratios for diffusion-axis coverage.

**The recovery loop is ~20 minutes.** At 100h ≈ 10M positions, `batch_tokens=16384`, 3 epochs →
~1,800 steps ≈ 0.09 EF ≈ 20 min on a single GPU. Use a *small* batch for experiments — at
production batch size, 100h gives you only ~200 steps.

That 20-minute loop means **treat the surgery configuration as something you sweep**, not something
you get right once.

**Start with English:** the teacher is strongest there (cleanest KD targets), the repo's eval
harness has `librispeech_pc` and `seedtts_en`, and there's no pinyin complication. Pull Emilia EN
shards — `run_emilia.sh` handles download → tokenization → training end to end.

**Calibration set must span the diffusion time axis.** `OmniVoiceSampleProcessor` samples
`mask_ratio ~ U(0,1)`, `prompt_ratio ~ U(0,0.3)`, and drops conditioning 10% of the time. Calibrate
at a single noise level and you get a biased subspace. Include the CFG unconditional branch's
audio-only topology as long as CFG is alive.

### Per-speaker policy

Because there's **no cross-utterance speaker mechanism**, per-speaker hours is a sampling knob.
What the objective consumes is **distinct prompt conditions** ≈ (speaker × session × style).

| Cap | Speakers @ 10,000h | @ 3,000h | Utts/speaker (15s avg) |
|---|---|---|---|
| 5 min | 120,000 | 36,000 | ~20 |
| **10 min** | **60,000** | **18,000** | **~40** |
| 15 min | 40,000 | 12,000 | ~60 |
| 30 min | 20,000 | 6,000 | ~120 |

**Cap 10–15 min, floor 45–60 s.** Two opposing forces:

- **Capping protects against prompt-ignoring.** If one voice has 50 hours the model can memorize
  it and generate that timbre from weak prompts — the exact failure mode you can't afford
- **Some depth aids timbre/prosody disentanglement.** If every speaker appears in one emotional
  state, the model can entangle timbre with delivery

10 min ≈ 40 utterances resolves this **provided the utterances span varied context**. So it's
really *sessions*, not minutes: 10 min of podcast is varied; 10 min of audiobook narration is one
state repeated. Prefer ≥2 distinct recordings per speaker.

**Cap per language against that language's quota, not globally in minutes.** If Polish yields
300h from 500 speakers, a flat 10-min cap discards 72% of your scarcest language.
`cap = max(10 min, whatever fills the quota)`, then log speaker count and top-1% share per language.

**Hold out speakers, not utterances,** for the dev set. Keep it 24 kHz only.

### Capability retention

The teacher supports 13 non-verbal tags — `[laughter]`, `[sigh]`, etc.
([`omnivoice.py:1528-1533`](OmniVoice/omnivoice/models/omnivoice.py#L1528-L1533)). The student
learns them only if its **transcripts contain those tags**. In-the-wild audio is full of laughter;
ASR transcripts never mark it. Same for `use_pinyin_ratio`, which requires a `text_pinyin` field.

**This is the one place teacher-generated data is clearly right** — a narrow capability transfer,
not a bulk corpus. Generate a few hundred hours of tagged utterances and fold them in as a small
slice. Decide deliberately; otherwise the capability silently disappears.

### Why NOT to use teacher-generated audio as the primary corpus

1. **Prompt distribution mismatch.** In training the reference is a prefix of the same utterance.
   If that utterance is synthetic, the prefix is synthetic — but at inference the reference is a
   real recording. For a model whose entire job is extracting identity from the prompt, that's the
   worst place for a domain gap.
2. **It cannot create diversity.** Cloning is conditioned on a reference, so synthetic speech has
   exactly as many voices as you have references. There's no emotion conditioning anywhere — only
   the reference's style and the text.
3. **It removes your regularizer.** The ground-truth CE term becomes vacuous and every teacher
   error becomes a training target.

Where it *is* right: Phase D trajectories (by construction), capability retention (above), and
targeted gap-filling capped at ~10–15% of any language's data.

---

## 9. Dataset map

### Sample rate findings ✅

The single most important data-quality result:

| Path | Token agreement | mel-L1 | Verdict |
|---|---|---|---|
| Native 24 kHz | 100% | 0.645 | codec floor |
| *24k→48k→24k (lossless control)* | *61.5%* | *0.675* | *— brittleness floor —* |
| **48k → 24k, proper resample** | **62.3%** | **0.696 (+3%)** | **safe ✅** |
| 48k → 24k, kaiser width=64 | 65.6% | 0.694 | no better |
| 48k → 24k, **naive `s[::2]`** | 42.5% | 0.890 | aliasing damage |
| **16k → 24k** | **26.9%** | **1.078 (+67%)** | limited use ⚠️ |
| 8k → 24k | 15.4% | 1.911 (+196%) | reject ❌ |

**Read the control row first.** An inaudible 48 kHz round-trip flips 38% of tokens while changing
audio by 5% — RVQ indices are extremely brittle, so token agreement massively overstates damage.
The mel-L1 column is the real signal.

**16 kHz is not useless.** The codec's semantic branch is HuBERT at `semantic_sample_rate: 16000`,
so a 16 kHz source loses *nothing* there — which is why codebook 0 survives at 53% while codebook 7
collapses to 3%. Only the acoustic branch degrades.

**The actual failure mode is correlation, not presence.** This is a clone model: the prompt sets
acoustic character, and a 16 kHz utterance yields a 16 kHz prompt, so the model learns
"band-limited prompt → band-limited output" — a *correct* mapping. What breaks it is correlation
with a conditioning variable. If Spanish is 100% MLS-derived, the student learns
**Spanish ⇒ dull**.

**Rules:**
1. Cap 16 kHz data at ~25–30% of the corpus
2. Never let it correlate with language — no language >50% 16 kHz-derived
3. Tag source bandwidth in metadata
4. Keep the dev set 24 kHz only
5. Reject 8 kHz outright

**Resampling:** the pipeline already does this correctly —
[`audio.py:84`](OmniVoice/omnivoice/utils/audio.py#L84) uses `torchaudio.functional.resample`,
which anti-aliases. Default `lowpass_filter_width=6` measured 0.696 vs 0.694 for kaiser width=64,
so **don't bother tuning it**; just don't hand-roll decimation.

**Clipping:** both proper resamplers produced peaks of **1.001** on a 0 dBFS source (filter
ringing). Modern loudness-maximized media sits at 0 dBFS, so **peak-normalize to ~−1 dBFS before
resampling**, or stay in float32 through tokenization.

**Nominal sample rate is the wrong filter — effective bandwidth is.** A "48 kHz" file from
low-bitrate Opus can be lowpassed at 8–12 kHz. Measure ✅:

```python
X = torch.stft(x, 1024, 256, window=torch.hann_window(1024), return_complex=True).abs()**2
freqs = torch.linspace(0, sr/2, X.shape[0])
hi = (X[freqs > 10000].sum() / X.sum()).item()
# genuine 24 kHz speech: ~3e-5      upsampled from 16 kHz: 0.0
# reject below ~1e-6
```

Run this as an audit over **every** corpus before tokenizing anything.

### Coverage by language

**Emilia / Emilia-Large** ([arXiv:2501.15907](https://arxiv.org/abs/2501.15907)) — 216k hours
(101k Emilia + 114k Emilia-YODAS), in-the-wild podcasts/interviews/talk shows, **24 kHz**, native
repo support via `run_emilia.sh` and `zhu-han/Emilia-Manifests`.
**Only 6 languages: En, Zh, De, Fr, Ja, Ko.** ⚠️ Believed CC-BY-**NC** — verify if commercial.

**YODAS2** ([HF](https://huggingface.co/datasets/espnet/yodas2),
[arXiv:2406.00899](https://arxiv.org/html/2406.00899v1)) — **24 kHz, CC-BY-3.0**, 140+ languages,
~420k hours (86k *manual* subtitles across 140 langs + 336k automatic). Long-form, not segmented.
**This is the universal answer for languages Emilia misses.** Prefer the manual-subtitle subset —
it sidesteps most misalignment. See also
[`espnet/yodas_owsmv4`](https://huggingface.co/datasets/espnet/yodas_owsmv4) — a 166k-hour
pre-filtered subset over 75 languages.

**CML-TTS** ([arXiv:2306.10097](https://arxiv.org/html/2306.10097),
[OpenSLR 146](https://www.openslr.org/146/)) — **24 kHz, CC-BY-4.0**, LibriVox-derived. But the
per-language breakdown is the problem:

| | Train hours | Speakers |
|---|---|---|
| German | 1128.96 | 168 |
| Dutch | 482.82 | 35 |
| **Spanish** | **279.15** | **77** |
| **Italian** | **73.78** | **61** |
| **Polish** | **30.61** | **8** |
| **Portuguese** | **23.14** | **30** |

Its bulk is German and Dutch. **Polish has 8 speakers** — worthless for zero-shot generalization.
Useful only as a small clean 24 kHz supplement for Spanish/Italian.

**MLS** ([OpenSLR 94](https://www.openslr.org/94/)) — **16 kHz**, CC-BY-4.0, audiobooks.
Spanish 1438.41h, Portuguese 284.59h, Italian 279.43h. Subject to the 16 kHz cap.

**Granary** ([arXiv:2505.13404](https://arxiv.org/pdf/2505.13404),
[HF](https://huggingface.co/datasets/nvidia/Granary)) — NVIDIA 2025, ~650k ASR hours, 25 European
languages, CC-BY-4.0. **Annotations only** — file paths point at YODAS / YouTube-Commons / MOSEL.
Treat it as a high-quality *label and filtering layer* over YODAS: their pipeline did segmentation,
two-pass inference, hallucination filtering, and punctuation restoration.

⚠️ **Composition matters more than size.** Polish is ~4.78M VoxPopuli rows against only 6.94k
YouTube-Commons rows — almost entirely parliamentary speech. Spanish has 862k YTC rows. For
prosodic diversity, Granary Polish is the wrong composition.

**Recommended stack for ES/IT/PL/PT:**

| Language | Primary | Supplement | Verdict |
|---|---|---|---|
| Spanish | YODAS2 `es` (24k) | CML-TTS 279h/77spk; MLS 1438h (16k); CV | Comfortable |
| Italian | YODAS2 `it` (24k) | CML-TTS 74h/61spk; MLS 279h (16k); CV | Workable |
| Portuguese | YODAS2 `pt` (24k) | MLS 285h (16k); CV | Workable |
| **Polish** | YODAS2 `pl` (24k) | MLS (small); VoxPopuli 111h; BIGOS; CV | **Bottleneck** |

Plan for Polish at 200–400h rather than 1,000.

### Diversity sources beyond Emilia

**Tier 1 — highest affect density. Small; upweight in the sampler.**

| | Why |
|---|---|
| **EARS** | ~107 speakers, emotions × speaking styles × freeform, **48 kHz anechoic**. Highest value per hour; no sample-rate caveat |
| **Expresso** | ~47h expressive English — whisper, laughing, sad, enunciated |
| **ESD** | 10 EN + 10 ZH speakers × 5 emotions, ~29h |
| **JVNV** | Japanese emotional with non-verbal vocalizations |

**Tier 2 — in-the-wild volume:** YODAS2, Emilia-YODAS, ReazonSpeech (~35k h Japanese TV — drama
and variety are unusually expressive), GigaSpeech, People's Speech.

**Tier 3 — speaker count:** VoxCeleb2 (~6k speakers of celebrity interviews; 16 kHz, no
transcripts, segments ~4–8s so **merge contiguous segments** from the same video), Common Voice
(~100k+ speakers but ~5s clips and flat read prosody), VoxPopuli.

**Tier 4 — audiobooks:** MLS, CML-TTS, Libri-Light, LibriTTS-R. Better prosody than reputation
(character voices, dramatic narration) but few speakers per language.

### Licensing ⚠️ verify all of these

| Dataset | License | Commercial? |
|---|---|---|
| YODAS / YODAS2 | CC-BY-3.0 | ✅ |
| CML-TTS | CC-BY-4.0 | ✅ |
| MLS | CC-BY-4.0 | ✅ |
| Granary | CC-BY-4.0 (annotations only) | ✅ |
| Common Voice | CC0 | ✅ |
| **Emilia / Emilia-Large** | believed **CC-BY-NC** | ⚠️ verify — it's 5–6 languages of your corpus |
| **CORAA** (pt-BR, 290h spontaneous) | **CC-BY-NC-ND 4.0** | ❌ **ND blocks derivative works** |

CORAA is otherwise ideal — 290.77h of manually validated *spontaneous* Brazilian Portuguese — but
**ND plausibly prohibits training on it**. Don't build Portuguese around it.

---

## 10. Filtering pipeline

### The framing that matters most

**Quality filtering and diversity selection are opposite operations and must be separate stages.**

- **Quality** = per-clip *reject* against thresholds
- **Diversity** = corpus-level *quota* over the survivors

The current `filter.py` conflates them — `min_total_duration_s: 300` is a corpus-level decision
implemented as a reject.

**And the interaction that will quietly ruin your corpus:** every quality metric is biased against
expressiveness. Shouting, whispering, laughing, and crying all score worse on DNSMOS and SNR than
calm narration. A global threshold is a prosody filter wearing a quality filter's clothes.

**Fix: apply quality thresholds within prosodic strata.** Bin by f0-variance × energy-dynamics ×
speaking-rate, then keep the top *N%* by DNSMOS *within each bin*. You get the best-recorded
shouting and the best-recorded whispering, not only the calm speech.

**Measure the bias first:** scatter DNSMOS against f0-std on a few thousand clips. The correlation
will be negative; its magnitude tells you what a global threshold costs.

### Changes to the existing `configs/dataset_filter.json`

| Field | Current | Change to | Why |
|---|---|---|---|
| `audio_length.min_ms` | 500 | **8000** | prefix-prompt scheme needs ≥8s (500ms → 150ms "reference") |
| `audio_length.max_ms` | 20000 | **30000** | `max_sample_tokens: 2000` allows 80s; chunking starts at 30s |
| `speakers.min_total_duration_s` | 300 | **30–60** | correct for single-speaker TTS; **backwards** for zero-shot cloning — the long tail is the valuable part |
| `metrics.dnsmos.min` | 3.6 | **3.0–3.2** | anti-correlated with expressiveness; you're distilling, so the bar is lower than from-scratch |
| `metrics.rms` | [0.04, 0.15] | normalize instead | `ref_rms` is explicitly modeled in `VoiceClonePrompt` |

(`prepare_dataset.py`'s Misaki G2P step isn't needed — OmniVoice consumes raw text.)

### The cascade — cheapest first

To retain 10k hours you'll scan ~100k. This ordering is the difference between hours and weeks.

| # | Stage | Cost | Catches |
|---|---|---|---|
| 0 | Duration window (8–30s), channels | free | wrong length |
| 1 | `EffectiveBandwidth`, `ClipRatio`, RMS, DC offset | >1000× RT/core | upsampled fakes, loudness-crushed sources |
| 2 | VAD speech ratio (Silero) | ~1000× RT | silence, sound effects, non-speech |
| 3 | **Audio tagging** (PANNs/YAMNet via onnxruntime) | ~300× RT | **music and singing** — essential for YODAS, usually skipped |
| 4 | `NoiseFloorSNR` from VAD silence | cheap | noisy field recordings |
| 5 | DNSMOS (relaxed) | ~200× RT | general quality |
| 6 | `IntraClipSpeakerVariance` — ECAPA sliding windows | ~500× RT | overlapping speakers, crosstalk |
| 7 | Language ID | cheap | YODAS's unreliable labels |
| 8 | **Transcript verification** — CTC forced-alignment, *not* full ASR | ~500× RT | subtitle misalignment |
| 9 | ASR (faster-whisper) | ~100× RT | transcripts where none exist |

Stages 8–9 dominate: ASR on 15k surviving hours at ~100× RT across 4 GPUs is ~37 h, versus ~6 h for
all of stage 1 on the full 100k. **Never run ASR before the cheap filters have cut the volume.**

For YODAS, prefer **forced-alignment scoring over re-transcription** — subtitles already exist, and
a small CTC model is ~5× cheaper than Whisper while directly measuring what's broken.

**Write every metric to a sidecar table** (parquet/JSONL keyed by clip id), not a pass/fail list.
Re-thresholding then becomes a pandas query instead of a re-scan.

### Corpus-specific gotchas

**YODAS:**
- **Music and singing** are the dominant contaminant — stage 3 is non-negotiable
- **Subtitle misalignment** is rampant in auto-generated captions
- ⚠️ **Synthetic narration** — YouTube is now full of AI-voiced content. Training a TTS model on
  TTS output is a real contamination risk with no clean mitigation
- Near-duplicate re-uploads — dedupe on audio fingerprint or transcript n-grams
- YODAS2 is long-form — VAD re-segment into 8–30s single-speaker chunks

**VoxCeleb2:**
- No transcripts — full ASR mandatory
- 16 kHz — subject to the cap
- Segments ~4–8s — **merge contiguous segments from the same video** (filenames encode video id +
  segment index). Skipping this discards most of the corpus
- Interview crosstalk and applause — stage 6 matters more here than anywhere
- **Upside: ground-truth speaker labels for ~6k speakers**, making the diversity stage trivial

### Diversity stage

Over the survivors, compute per clip: f0 median/std/range (`librosa.pyin`), speaking rate (ASR word
count ÷ speech duration), energy dynamics (RMS std), pause fraction, duration. Then:

1. **Per-speaker cap** — ground-truth labels for VoxCeleb2, ECAPA clustering for YODAS
2. **Stratified sampling in prosodic space** — normalize, k-means into ~100 cells, sample equally
   (or ∝ √population). Random sampling from a big corpus just reproduces its mode
3. **SER as an extra axis** — raw in-the-wild data is ~80–90% neutral; sampling toward 50/50 costs
   nothing and roughly doubles affect range
4. **Log what you dropped**, per stage and per language — silent attrition is how you end up with
   4,000h Spanish and 200h Polish without noticing

### Metrics to add to `src/metrics.py`

All fit the existing `__call__(audio) -> float` protocol except the last:

`EffectiveBandwidth`, `ClipRatio`, `SpeechRatio`, `NoiseFloorSNR`, `IntraClipSpeakerVariance`,
`ProsodyStats` (returns a dict).

Already available in the environment: `speechbrain` 1.1.0 (+ `spkrec-ecapa-voxceleb` cached),
`faster-whisper` 1.2.1, `librosa`, `onnxruntime` (DNSMOS). Only the stage-3 audio tagger is missing.

---

## 11. Measured reference values

### 11.1 End-to-end inference benchmark ✅

**Hardware:** Apple M4 Pro, 24 GB, 14 cores. Teacher (612M) on **MPS fp16**; codec on CPU fp32
(the codec cannot run on MPS — output channels > 65536,
[`omnivoice.py:266-271`](OmniVoice/omnivoice/models/omnivoice.py#L266-L271)).
Reference 6.0s → 145 frames. Timings synchronised with `torch.mps.synchronize()`.

> **Setup note:** `from_pretrained(device_map="mps", dtype=float16)` **segfaults** (exit 139)
> during weight loading with torch 2.13.0 + transformers-from-git. Workaround: load with
> `device_map="cpu"`, detach `audio_tokenizer`, `.to("mps", torch.float16)`, then reattach the
> codec so it stays on CPU. Also `torchaudio.load` fails here (torchcodec can't find ffmpeg
> libs) — use `soundfile`.

**num_step=32** — total 7.26s for 9.02s audio, **RTF 0.805**

| Component | sec | % total | calls | ms/call |
|---|---|---|---|---|
| **LM forward** | 7.005 | **96.5%** | 32 | 218.9 |
| sampler scoring (CFG + gumbel) | 0.068 | 0.9% | 32 | 2.1 |
| sampler loop overhead (Python) | 0.006 | 0.1% | | |
| codec decode (CPU) | 0.169 | 2.3% | 1 | 169.3 |
| input prep / tokenize | 0.002 | 0.0% | 1 | 1.9 |
| other (postproc, transfers) | 0.011 | 0.2% | | |

**num_step=16** — total 3.73s for 9.11s audio, **RTF 0.409**. LM forward 94.1%, codec decode 4.5%.

**Conclusion: the LM forward is essentially the entire pipeline.** Everything else — codec decode,
scoring, tokenization, post-processing — sums to under 4%. Optimization effort belongs entirely on
the LM.

### 11.2 Batch scaling — the Python loop is not a bottleneck ✅

16 steps, B=1/4/8:

| B | total s | LM s | scoring s | Python loop s | **loop %** | s/item |
|---|---|---|---|---|---|---|
| 1 | 2.82 | 2.68 | 0.025 | 0.004 | **0.1%** | 2.82 |
| 4 | 10.69 | 10.14 | 0.103 | 0.009 | **0.1%** | 2.67 |
| 8 | 21.30 | 20.19 | 0.208 | 0.018 | **0.1%** | 2.66 |

This **refutes** the earlier claim that vectorizing the per-item loop was the highest-value fix.
Note also that `s/item` barely improves with batch (2.82 → 2.66), i.e. batching buys almost nothing
on MPS — the device is already saturated at B=1.

### 11.3 Forward-time scaling — linear, not quadratic ✅

Raw teacher forward, MPS fp16, synthetic inputs:

| S | B=1 (ms) | B=2 / CFG (ms) | ratio | **ms per token** |
|---|---|---|---|---|
| 256 | 66.0 | 126.4 | 1.91× | 0.258 |
| 384 | 94.7 | 188.0 | 1.98× | 0.247 |
| 512 | 127.3 | 251.5 | 1.97× | 0.249 |
| 768 | 191.6 | 390.3 | 2.04× | 0.249 |
| 1024 | 262.3 | 534.0 | 2.04× | 0.256 |

**Two results.** First, **ms/token is flat** across a 4× range of S — the attention S² term is not
visible in wall-clock at these lengths; the forward is dominated by the linear projection+MLP term.
(The slight uptick at S=1024 is the quadratic term starting to appear.) This means sliding-window
attention would gain even less than the 1.06× estimated in §5.5, and confirms that the prefix
K/V blocking win is mostly about skipping the prefix's *projections and MLP*, not its attention.

Second, the **B=2 ratio of ~2×** is the direct measurement behind the CFG correction in §5.

### 11.4 Cost amortization ✅

16 steps, B=1, varying target length:

| target words | audio s | ms/forward | RTF |
|---|---|---|---|
| 8 | 2.72 | 127.8 | 0.785 |
| 20 | 6.78 | 183.0 | 0.457 |
| 40 | 13.23 | 277.3 | 0.361 |

RTF improves with longer outputs because the fixed prefix cost amortizes over more generated audio.
This is the same effect that makes **prefix K/V blocking most valuable for short outputs**
(§5.2: 2.20× at P=150/T=125 vs 1.40× at P=300/T=750).

### 11.5 Static reference values



Quick-reference table of everything measured against the local checkpoint. ✅

| Quantity | Value |
|---|---|
| Teacher params | 612.58M (313 tensors) |
| Codec params | 201.40M (527 tensors); decode path **21.56M** |
| Codec decode, 10s, CPU 8 threads | 181 ms, RTF 0.0181, 55× realtime |
| Codec encode, 10s, CPU 8 threads | 275 ms, RTF 0.0275, 36× realtime |
| Frame rate / sample rate | 25 Hz / 24,000 Hz |
| Bitrate | 2 kbps (8 × 10 bits × 25 Hz) |
| Codec round-trip mel-L1, 8/6/4/2 codebooks | 0.638 / 0.698 / 0.774 / 1.03 |
| Native 24k >8 kHz energy | ~1.2×10⁻⁴ |
| 16k-upsampled >8 kHz energy | **exactly 0.0** |
| mel-L1: native / lossless-control / 48k / 16k / 8k | 0.645 / 0.675 / 0.696 / 1.078 / 1.911 |
| Resample peak on 0 dBFS source | 1.001 (clips) |
| `t_shift` max per-step: n=32@0.1 vs n=16@0.2 | 0.244 vs 0.250 |
| Qwen3 tokenizer by script (Latin/CJK/mixed) | 76054 / 25464 / 19258 of 151676 |
| Prefix blocking, P=200 T=250 N=16 | 135.5G → 78.7G = **1.72×** |

**Note on the tokenizer histogram:** script-based vocab pruning for a typical 10-language set
retains ~144k of 151k tokens — essentially nothing. **Frequency-based pruning is required.**

---

## 12. Constraints and dead ends

**Hard constraints:**

- **Do not modify or retrain the codec.** Every teacher token, every cached trajectory, and the
  teacher's entire 8×1025 output space are defined by these exact codebooks. Reducing the codebook
  *count* is safe (prefix truncation of the same RVQ); anything else is not.
- **Frame rate is codec-locked** at 25 Hz (hop 960 @ 24 kHz).
- **GQA requires `H % Kv == 0`** with equal group sizes.

**Doesn't work:**

- **KV *merging*** (Kv 8→4 at H=16) — saves 16.7% of attention params but **zero** S² cost, since
  attention scales with query heads only. And there's no KV cache here, so GQA's usual payoff is
  absent. Prune groups instead.
- **Fewer codebooks to shorten the sequence** — codebooks are summed into one embedding per frame,
  so S is frames only.
- **Linear attention** — no teacher to distill from, and at S=450 the S² term isn't where the money is.
- **Token merging in the sequence** — output must be exactly 25 Hz.

**Evaluation discipline:**

Evaluate end-to-end throughout, never on layer-wise reconstruction error. 16 sequential steps
compound: 0.99 per-step cosine similarity can still drift. Use `fleurs` (multilingual WER),
`seedtts_en`/`seedtts_zh`, speaker-SIM, and UTMOS from [`omnivoice/eval/`](OmniVoice/omnivoice/eval/).
Once CFG is distilled away, run evals with `guidance_scale=0`.

---

## 13. Recommended order of work

### Immediately — zero training data

1. **`t_shift=0.2` sweep on the teacher at 16 steps.** Zero data, minutes. Tells you how much of
   the 32→16 gap is schedule shape vs. model capacity, which sets how hard Phase D has to work.
2. **Prefix-blocking mask on the teacher.** Zero training. Validates a 1.72× before you commit.

These two are the highest value-per-hour in the plan — pure measurement on the existing teacher,
de-risking 2× and 1.72× of the stack between them.

### Next — implementation unblocking

3. **Raise `batch_tokens`** and re-tune `steps`/LR. (Sampler-loop vectorization was previously
   listed here; measurement in §11.1 shows it is 0.1% of inference time — deprioritized.)

### Then — ~5 hours of audio

5. Phase B profiling: residual eigenspectrum (go/no-go on width), block influence (which layers),
   FFN neuron scores, attention locality per layer.
6. **Codebook 8 vs 6 ablation** with UTMOS.
7. Surgery + least-squares repair.
8. **Measure init quality before any training.** If PCA init doesn't beat magnitude-pruning init at
   step 0, you've learned that in an afternoon.

### Then — 50–200 hours, English

9. Recovery experiment, ~20 min/run. Sweep surgery configs.

### Finally — full corpus

10. Phase A (vocab), Phase C (KD at 32 steps), Phase D (steps + guidance), Phase E (sampler re-tune).

---

### Go/no-go checks

- **Residual eigenspectrum:** if the top 704 of 1024 directions capture <99% of energy, skip the
  width reduction and put the budget into depth + vocab + repair, which degrade more gracefully.
- **Post-surgery loss on held-out data, before training:** the single most informative number, and
  it's free.
- **Teacher at 16 steps with `t_shift=0.2`:** if quality holds, Phase D is easy; if not, budget for
  on-policy refinement.

### Closing note

You are stacking ~3× parameter reduction, 2× step reduction, 2× CFG removal, and 1.72× prefix
blocking. Each is individually reasonable; together they compound, and the last one you add will
look like the culprit even when it isn't. The staged gates exist so you can attribute regressions.

The one thing genuinely working in your favor is the **language cut** — going 646 → 10 frees a
large amount of real capacity, and it's the reason a 3× compression can plausibly land near teacher
quality rather than well below it.

**Your real budget is engineering time, not GPU time.** Eight components to build, and the compute
per experiment is small enough that you'll spend far more time writing surgery and KD code than
running it.
