#!/usr/bin/env python3
"""Prefix K/V caching: run the prefix once, then only the target each step.

`_generate_iterative` recomputes the whole sequence on every diffusion step even
though the prefix (`style | text | ref_audio`) never changes -- it has to, because
full bidirectional attention makes the prefix's hidden states depend on the
target. Cutting prefix-query -> target-key attention removes that dependency, and
then the prefix's K/V is constant and can be computed once.

This module does the actual hoist, which `--prefix-blocked` (mask only) does not:

    once:      forward the prefix, keep per-layer K/V           (P positions)
    per step:  forward ONLY the target against that cache       (T positions)

It also drops the unconditional branch's padding. That branch is built target-only
(`omnivoice.py:1342`) but lives in the same `(2B, C, max_c_len)` tensor and is
padded to the conditional length, so ~58% of its compute is spent on padding that
prefix blocking alone cannot reach.

Implemented by wrapping `forward`, not by forking the sampler: the loop reads only
`logits[i, :, c_len-t_len:c_len]` and `logits[B+i, :, :t_len]`, so returning a
tensor with just those regions filled is indistinguishable to it.

Correctness is checked, not assumed -- `verify()` compares against the mask-only
path, which must agree to float tolerance.
"""

import torch
from transformers.cache_utils import DynamicCache


class PrefixCachedGenerator:
    """Wraps `model.forward` so generation reuses a cached prefix.

    Items are processed one at a time. `c_len`/`t_len` differ per item, so a
    batched implementation would need padding and its own mask bookkeeping --
    the wins here are per-generation, and B=1 is the interactive case that
    matters. B>1 still works, just without cross-item batching.
    """

    def __init__(self, model):
        self.model = model
        self._orig_forward = model.forward
        self._orig_iter = model._generate_iterative
        self.caches = {}
        self._buf = None
        self.prefix_calls = 0
        self.target_calls = 0

        def generate_iterative(task, gen_config):
            self.caches = {}          # a new generation invalidates everything
            self._buf = None
            return self._orig_iter(task, gen_config)

        model._generate_iterative = generate_iterative
        model.forward = self.__call__

    def restore(self):
        self.model.forward = self._orig_forward
        self.model._generate_iterative = self._orig_iter

    # -- the LM call --------------------------------------------------------

    def _run(self, emb, position_ids, attn, past=None):
        out = self.model.llm(
            inputs_embeds=emb,
            attention_mask=attn,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=past is not None,
            return_dict=True,
        )
        return out.last_hidden_state

    def _heads(self, hidden):
        """[1, L, H] -> [1, C, L, V], matching OmniVoice.forward's layout."""
        cfg = self.model.config
        b, L, _ = hidden.shape
        return (self.model.audio_heads(hidden)
                .view(b, L, cfg.num_audio_codebook, cfg.audio_vocab_size)
                .permute(0, 2, 1, 3))

    def __call__(self, input_ids=None, audio_mask=None, attention_mask=None, **kw):
        am = attention_mask
        if am is None or am.dim() != 4 or am.shape[0] % 2 != 0:
            return self._orig_forward(input_ids=input_ids, audio_mask=audio_mask,
                                      attention_mask=am, **kw)

        model, cfg = self.model, self.model.config
        B = am.shape[0] // 2
        dev = input_ids.device

        # `_prepare_embed_inputs` is positionwise, so embed only the slices that are
        # actually forwarded rather than the whole padded sequence every step.
        def embed(row, lo, hi):
            return model._prepare_embed_inputs(input_ids[row:row + 1, :, lo:hi],
                                               audio_mask[row:row + 1, lo:hi])

        S = input_ids.shape[-1]
        dtype = model.audio_heads.weight.dtype
        shape = (2 * B, cfg.num_audio_codebook, S, cfg.audio_vocab_size)
        # Reused across steps: both read regions are fully overwritten every call,
        # so re-zeroing a ~13 MB tensor 16 times per generation buys nothing.
        if self._buf is None or tuple(self._buf.shape) != shape:
            self._buf = torch.zeros(shape, device=dev, dtype=dtype)
        logits = self._buf

        for i in range(B):
            # Max over query rows, not row 0: target queries are never masked and
            # carry the full count, so this stays correct even if a mask-blocking
            # wrapper has already edited the tensor in place.
            c_len = int(am[i, 0].sum(-1).max())
            t_len = int(am[B + i, 0].sum(-1).max())
            p = c_len - t_len
            if not 0 < p < c_len:                    # no prefix: nothing to cache
                return self._orig_forward(input_ids=input_ids, audio_mask=audio_mask,
                                          attention_mask=am, **kw)

            # ---- conditional branch: cached prefix + target-only forward ----
            if i not in self.caches:
                cache = DynamicCache()
                pos = torch.arange(p, device=dev).unsqueeze(0)
                mask = torch.ones(1, 1, p, p, dtype=torch.bool, device=dev)
                self._run(embed(i, 0, p), pos, mask, cache)
                self.caches[i] = cache
                self.prefix_calls += 1

            cache = self.caches[i]
            assert cache.get_seq_length() == p, (
                f"cache is {cache.get_seq_length()} long, expected {p}")

            pos = torch.arange(p, c_len, device=dev).unsqueeze(0)
            # Target queries see the whole prefix and the whole target.
            mask = torch.ones(1, 1, t_len, c_len, dtype=torch.bool, device=dev)
            hidden = self._run(embed(i, p, c_len), pos, mask, cache)
            # Negative = remove that many tokens. A positive argument means
            # "absolute final size" in transformers <5.18 and is deprecated there,
            # so the negative form is the one that keeps working either way.
            cache.crop(-t_len)                      # drop the target, keep the prefix
            self.target_calls += 1
            logits[i, :, p:c_len, :] = self._heads(hidden)[0]

            # ---- unconditional branch: target-only, padding dropped ----
            pos = torch.arange(t_len, device=dev).unsqueeze(0)
            mask = torch.ones(1, 1, t_len, t_len, dtype=torch.bool, device=dev)
            hidden = self._run(embed(B + i, 0, t_len), pos, mask)
            logits[B + i, :, :t_len, :] = self._heads(hidden)[0]

        from omnivoice.models.omnivoice import OmniVoiceModelOutput
        return OmniVoiceModelOutput(loss=None, logits=logits)


def enable(model):
    return PrefixCachedGenerator(model)


def verify(model, prompt, text, language="en", num_step=4, atol=2e-2):
    """Cached path vs mask-only path: the read regions must agree.

    Both compute the same function, so any disagreement beyond float noise means
    the cache, the position ids, or the mask is wrong.

    The slice bounds are computed from the public preparation path rather than
    read back off the mask -- `enable_prefix_blocking` edits the mask in place, so
    a wrapper that inspects it afterwards recovers the prefix length, not `c_len`.
    """
    import numpy as np
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    from generate_samples import enable_prefix_blocking

    t_len = int(model.duration_estimator.estimate_duration(
        text, prompt.ref_text, prompt.ref_audio_tokens.shape[1]))
    c_len = model._prepare_inference_inputs(
        text, t_len, prompt.ref_text, prompt.ref_audio_tokens,
        language, None, True)["input_ids"].size(2)
    print(f"  c_len={c_len}  t_len={t_len}  prefix={c_len - t_len}")

    cfg = OmniVoiceGenerationConfig(num_step=num_step)
    grabbed = {}

    def capture(tag, orig):
        def fn(*a, **kw):
            out = orig(*a, **kw)
            grabbed.setdefault(tag, []).append(
                out.logits[0, :, c_len - t_len:c_len, :].float().cpu().numpy())
            return out
        return fn

    # arm 1: mask-only blocking (the reference -- same maths, no cache)
    orig = enable_prefix_blocking(model)
    model.forward = capture("mask", model.forward)
    torch.manual_seed(0)
    model.generate(text=text, language=language, voice_clone_prompt=prompt,
                   generation_config=cfg)
    model.forward = orig

    # arm 2: cached prefix
    gen = enable(model)
    model.forward = capture("cache", model.forward)
    torch.manual_seed(0)
    model.generate(text=text, language=language, voice_clone_prompt=prompt,
                   generation_config=cfg)
    gen.restore()

    a, b = grabbed["mask"], grabbed["cache"]
    n = min(len(a), len(b))
    worst = max(float(np.abs(a[k] - b[k]).max()) for k in range(n))
    corr = float(np.corrcoef(a[0].ravel(), b[0].ravel())[0, 1])
    agree = float(np.mean(a[0].argmax(-1) == b[0].argmax(-1)))
    print(f"  {n} steps compared   max|delta| {worst:.5f}   "
          f"corr {corr:.6f}   argmax agreement {agree*100:.2f}%")
    print(f"  prefix forwards {gen.prefix_calls}, target forwards {gen.target_calls}")
    return worst <= atol
