# Prefix K/V Blocking

**Stage 2 of the distillation plan.** How to stop recomputing 58% of the sequence on every
diffusion step, why it is safe, and what it is actually worth on this checkpoint.

Provenance markers, matching the other docs in `knowledge/`:
✅ measured on this repo · 📊 computed from measured values · ⚠️ unverified

---

## 1. The one-sentence version

The reference audio and the text prompt never change during generation, but because attention
is fully bidirectional they are re-encoded from scratch at every one of the 16–32 steps —
**block the prefix from attending to the target, and the prefix becomes a static K/V cache
computed once.**

---

## 2. The problem

A voice-clone sequence is laid out as:

```
[  style  |  text  |  ref_audio  |  target_audio  ]
 <------------ prefix ---------->   <-- changes -->
        171 tokens ✅                 126 tokens ✅
```

Those figures are measured at the benchmark operating point (`runs/benchmark`, 25 held-out dev
pairs, ~5 s reference → ~5 s target): **P = 171.4, T = 125.8, S = 297.3** — so the prefix is
**58% of every forward pass.** The `S = 298` here matches the sequence length the backbone
actually reported during benchmarking, so this is the real geometry, not an estimate.

Now look at the sampling loop ([`omnivoice.py:1385`](omnivoice/models/omnivoice.py#L1385)).
Each step calls the model on the *entire* batch:

```python
batch_logits = self(input_ids=batch_input_ids, ...).logits
```

and then writes back **only** the target slice
([`omnivoice.py:1424-1425`](omnivoice/models/omnivoice.py#L1424-L1425)):

```python
batch_input_ids[i, :, c_len - t_len : c_len] = sample_tokens   # target only
```

The prefix token *ids* are identical on step 16 and step 1. Yet the model recomputes their
embeddings, all 28 layers of attention and MLP, and their output logits — **and then discards
those logits**, because only `c_len - t_len : c_len` is ever read.

At 16 steps that is 171 × 15 ≈ **2,570 token-forwards of pure waste per generation**, in a
pipeline where the LM is **93.9% of wall-clock time** ✅.

---

## 3. Why you cannot simply cache it

The obvious fix — run the prefix once, keep its K/V, reuse it — does not work on this model
as written, for one reason: **attention is fully bidirectional.**

The collator builds a mask where *every* query attends to *every* non-padding key
([`collator.py:92-94`](omnivoice/data/collator.py#L92-L94)):

```python
attention_mask = valid[:, None, None, :].expand(B, 1, max_len, max_len)
```

and inference does the same ([`omnivoice.py:1339`](omnivoice/models/omnivoice.py#L1339)):

```python
batch_attention_mask[i, :, :c_len, :c_len] = True
```

There is no causal mask anywhere, and `self.llm(...)` is called with no `past_key_values` and
no `use_cache` ([`omnivoice.py:522`](omnivoice/models/omnivoice.py#L522)) — **the model has no
KV cache at all.**

So prefix *queries* attend to target *keys*. The target changes every step, therefore the
prefix's hidden states change every step, therefore its K/V genuinely differ every step. The
recomputation is not laziness; it is required by the current attention topology.

**The waste is caused by an information flow that is never used.**

---

## 4. The idea

Cut exactly one quadrant of the attention matrix:

```
                          K E Y S
                ┌───────────────┬──────────────┐
                │    prefix     │    target    │
      ┌─────────┼───────────────┼──────────────┤
  Q   │ prefix  │       ✓       │      ✗       │  ← the only change
  U   ├─────────┼───────────────┼──────────────┤
  E   │ target  │       ✓       │      ✓       │
  R   └─────────┴───────────────┴──────────────┘
  I
  E S       "the target still sees everything;
             only the reverse direction is cut"
```

The prefix now attends only to itself, so its hidden states depend on nothing that changes.
Generation becomes:

```
once:        forward the prefix (171 tokens)  ->  stash K/V for all 28 layers
each step:   forward only the target (126 tokens), attending to [cached prefix K/V | target K/V]
```

This is the **prefix-LM / encoder-decoder split**: the prefix is an encoder run once, the
target is a decoder run repeatedly.

---

## 5. Why it is sound

Three independent arguments, all checkable in the code.

**5.1 The prefix is never supervised.** In `OmniVoiceSampleProcessor.__call__`, style and text
labels are set to `-100` outright, and the prompt region of the audio is too
([`processor.py:111`](omnivoice/data/processor.py#L111),
[`processor.py:126`](omnivoice/data/processor.py#L126),
[`processor.py:148`](omnivoice/data/processor.py#L148)):

```python
style_labels = torch.full(style_inputs.shape, -100)   # Style prompt does not compute loss
text_labels  = torch.full(text_inputs.shape, -100)    # Text does not compute loss
audio_labels[:, :prompt_length] = -100                # No loss on prompt region
```

The prefix's output logits have **never** been trained to mean anything. Its entire job is to
supply keys and values to the target. Blocking removes a signal path into a set of predictions
that are thrown away.

**5.2 The prefix is fully observed.** Masking is applied only to `audio_tokens[:, prompt_length:]`
([`processor.py:142`](omnivoice/data/processor.py#L142)). The prefix contains clean
ground-truth tokens — nothing there is noisy, so nothing there needs the target's help to be
denoised. Contrast the target, which is mostly `audio_mask_id` and genuinely does need context.

**5.3 The direction that matters is preserved.** Voice cloning requires
`target ← prefix` (timbre and text must reach the generated audio). That edge is untouched.
Only `prefix ← target` is cut, and the model has no objective that depends on it.

---

## 6. What it actually costs

It is not free, and it should not be described as free.

Bidirectional attention lets the prefix's representation be *contextualised by the target* —
the reference audio's encoding can, in principle, specialise toward what is being generated.
Blocking removes that. The prefix's K/V become a fixed, target-agnostic encoding.

So this **changes the function the model computes**, and the weights were trained under the
other topology. That is why stage 2 involves retraining rather than a config flag:
teacher runs with full attention, student with blocked attention, student learns to match.

⚠️ The docstring of [`scripts/train_prefix_blocked.py`](scripts/train_prefix_blocked.py) claims
applying the blocked mask to the *unmodified* teacher costs **+0.013 weighted CE (99.6% of
information retained)** — i.e. the model is already nearly invariant to the change. There is no
log or report in `runs/prefix_blocked/` backing that number, and CE is not the metric that
decides this. **Reproduce it, and measure WER, before relying on it.**

---

## 7. Technical implementation

**It is a mask change, not architecture surgery.** No parameters are added, removed, or
reshaped; the checkpoint stays a stock Qwen3.

**Inference.** The mask is already built explicitly, position by position, at
[`omnivoice.py:1330-1348`](omnivoice/models/omnivoice.py#L1330-L1348). Blocking is one extra
line — zero the prefix-query/target-key block:

```python
batch_attention_mask[i, :, :c_len, :c_len] = True
batch_attention_mask[i, :, : c_len - t_len, c_len - t_len : c_len] = False   # prefix -/-> target
```

That alone validates the quality question with **zero training**. The *speedup* then requires
the real change: hoisting the prefix forward out of the step loop and threading a per-layer K/V
cache into the target-only passes.

**Training, SDPA path.** [`PaddingDataCollator`](omnivoice/data/collator.py#L33) already emits
a 4D `[B, 1, L, L]` boolean mask, so the same two-line edit applies.

**Training, flex_attention path.** `_mask_mod_packed`
([`omnivoice.py:1464`](omnivoice/models/omnivoice.py#L1464)) currently encodes document
membership only:

```python
def _mask_mod_packed(document_ids, b, h, q_idx, kv_idx):
    same_doc = document_ids[q_idx] == document_ids[kv_idx]
    return same_doc
```

A blocked variant additionally needs a per-position `is_target` vector, and returns
`same_doc & (is_target[q_idx] | ~is_target[kv_idx])`.

**The one real subtlety: the prefix boundary must be defined identically in training and
inference.** At inference the prefix is `style + text + ref_audio` and the target is the
trailing `t_len` positions. In training the split is `prompt_length = int(T * prompt_ratio)`
with `prompt_ratio ~ U(0, 0.3)` ([`processor.py:82`](omnivoice/data/processor.py#L82)) — a
*random* boundary that moves every sample. Train with a boundary distribution that does not
cover the inference-time ratio and the student learns the wrong topology.

---

## 8. What it is worth 📊

Token-forwards at the measured geometry (P = 171.4, T = 125.8, N = 16). Because §11.3 of the
plan measured **ms/token as flat across sequence length**, the linear projection+MLP term
dominates wall-clock here, so token-forwards are a good proxy — attention's `S²` term works out
to a near-identical ratio anyway, since it is the *query* count that shrinks.

| | token-forwards | speedup |
|---|---|---|
| Full attention, conditional branch | 4,756 | 1.00× |
| **Blocked, conditional branch** | **2,185** | **2.18×** |

But with CFG on — which is the current default — the picture changes sharply:

| | token-forwards | speedup |
|---|---|---|
| Full attention, both branches | 9,513 | 1.00× |
| **Blocked, CFG as implemented today** | **6,941** | **1.37×** ⚠️ |
| Blocked, CFG with the unconditional branch trimmed | 4,198 | **2.27×** |

**Prefix blocking is throttled from 2.18× to 1.37× by something unrelated to it.** The
unconditional branch has *no prefix at all* — it is built as target-only
([`omnivoice.py:1342`](omnivoice/models/omnivoice.py#L1342)):

```python
batch_input_ids[B + i, :, :u_len] = inp["input_ids"][..., -u_len:]
```

— but it lives in the same `(2B, C, max_c_len)` tensor and is therefore **padded to the full
conditional length**, with pad positions masked to attend only to themselves. So 58% of the
unconditional branch's compute is spent on padding that prefix blocking cannot touch.

Two independent ways to recover it, and they lead to the same place:

1. **Trim the unconditional branch** — an implementation change, no retraining. ⚠️ It costs the
   batched forward (two different lengths), which on MPS is close to free (§11.2 measured
   batching buying almost nothing) but may not be on the RTX 6000. Measure before assuming.
2. **Stage 3, guidance distillation** — the unconditional branch disappears entirely, and
   prefix blocking delivers its clean 2.18×.

---

## 9. How it composes with the rest of the plan

- **Stage 1 (16 steps) — already done, and it makes this *relatively* more valuable.** The
  one-time prefix pass amortises over fewer steps, so blocking's share of the remaining work
  goes up slightly (2.27× at N=32 → 2.18× at N=16 for the conditional branch — the ratio dips
  because the fixed prefix pass is spread thinner, but the absolute saving per generation is
  what you bank).
- **Stage 3 (CFG distillation) — strongly synergistic**, per §8: it removes the padding that is
  currently eating a third of this stage's benefit.
- **Stages 4–5 (pruning, step distillation)** — orthogonal. This is a change to attention
  topology; those change parameters and schedule.
- **Short outputs benefit most.** The prefix is a fixed cost, so the win grows as the target
  shrinks — the plan's own table spans 2.20× at P=150/T=125 down to 1.40× at P=300/T=750. Our
  ~5 s/~5 s operating point sits near the favourable end, which is worth remembering when
  quoting the number: **it is geometry-dependent, not a property of the model.**

---

## 10. How to validate

In order, cheapest first:

1. **Zero-shot quality.** Apply the blocked mask to the unmodified teacher (§7, two lines) and
   run `scripts/step_reduction/wer_sweep.py`. Baseline to beat: **0.00% WER on `data/test`** at
   both 32 and 16 steps. If the blocked teacher holds 0.00%, the retraining in
   `runs/prefix_blocked/` may be unnecessary.
2. **Blind listening.** 9 pairs from `data/test`, as in stage 1 — WER cannot see micro-pauses,
   prosody, or timbre drift, and that is precisely where a changed conditioning path would show.
3. **Real speedup.** The 2.18× above is a token count, not a measurement. Implement the K/V
   hoist and re-run `scripts/benchmark.py` at the same operating point (**baseline: RTF 0.496 at
   16 steps, 1.02× realtime at 32**). Cache-management overhead and smaller-matmul efficiency are
   not in the arithmetic.
4. **Only then retrain**, if 1–2 show a gap worth closing.

The ordering matters because step 1 costs minutes and may make steps 3–4 unnecessary — the same
shape of result as stage 1, where the recommended tuning turned out to be the entire problem.

---

## 11. `generate()` did not apply the block — and silently produced garbage ✅

`OmniVoice._generate_iterative` built the conditional mask as
`batch_attention_mask[i, :, :c_len, :c_len] = True` — **full bidirectional attention**. Every
stage-2 descendant (`models/p2`, `models/p4/*`, `models/p4+p3/*`) is trained and deployed with
prefix → target attention removed, so the stock generation path ran all of them under the wrong
topology.

The failure is not subtle but it is easy to misread: the model speaks part of the **reference
transcript** before the target text, and drops or garbles target words. It looks like a bad
checkpoint or a duration-estimator bug. Measured on `diana`, `round_07_tuned_with_kd`, 16 steps:

| | w=2.0 | w=1.0 |
|---|---|---|
| full attention (stock) | 16.0% WER | 32.0% WER |
| prefix blocked | **0.0%** | **0.0%** |

Across the 9-clip `data/test` set after the fix: mean WER 0.8% (teacher, CFG) and 1.2% (student,
no CFG), with 8 of 9 clips at exactly 0.0%.

Fixed by adding `prefix_blocked: bool = False` to `OmniVoiceGenerationConfig` and applying the
block to the conditional rows. It is **opt-in**: the original `k2-fsa/OmniVoice` was not trained
with the block and must not get it. The unconditional rows are target-only, so they have no prefix
and need no block.

**This omission has now occurred three separate times** (a WER sweep, a duration test, and a blind
test), each time producing a plausible-looking but invalid result. Topology mismatch remains the
largest single degradation measured in this project. **Any generation from a stage-2 descendant
must pass `prefix_blocked=True`, and any result produced without it should be discarded, not
interpreted.**
