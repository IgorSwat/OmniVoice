"""Vocab pruning: shrink ``llm.embed_tokens`` to the tokens the corpus needs.

Implements "Phase A — vocab pruning" of ``knowledge/distilation_plan.md``.

The text embedding is untied -- the checkpoint carries no ``lm_head`` and
``audio_heads`` is the only output -- so rows for tokens the tokenizer never
emits are exactly dead weight.

Qwen2 BPE assigns token id ``256 + merge rank`` and sets ``ignore_merges:
false``, so the vocabularies a BPE tokenizer can actually reach are exactly the
sets closed downward under the merge graph. A prefix cutoff is the crudest such
set; the closure of the tokens the corpus uses fits the same budget while
reproducing corpus tokenization exactly. Keeping all 256 byte tokens means
unseen text re-splits into smaller pieces rather than going out of vocabulary.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from omnivoice.utils.text import add_punctuation

EMBED_KEY = "llm.embed_tokens.weight"
N_BYTE_TOKENS = 256
STYLE = "<|lang_start|>{lang}<|lang_end|><|instruct_start|>None<|instruct_end|>"


def read_transcripts(path: Path):
    csv.field_size_limit(10**7)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="|"):
            text = (row["transcription"] or "").strip()
            if text:
                yield f"<|text_start|>{add_punctuation(text)}<|text_end|>", row["language"]


def count_tokens(path: Path, tokenizer: Tokenizer, batch_size: int = 2000) -> Counter:
    """Token frequencies over the transcripts, in the processor's own formatting."""
    counts: Counter = Counter()
    batch: list[str] = []
    languages = {"None"}

    def flush() -> None:
        for encoding in tokenizer.encode_batch_fast(batch):
            counts.update(encoding.ids)
        batch.clear()

    for text, language in read_transcripts(path):
        batch.append(text)
        languages.add(language)
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()

    for language in languages:
        for denoise in ("", "<|denoise|>"):
            counts.update(tokenizer.encode(denoise + STYLE.format(lang=language)).ids)
    return counts


def merge_parents(model: dict) -> dict[int, tuple[int, int]]:
    vocab = model["vocab"]
    parents = {}
    for rank, merge in enumerate(model["merges"]):
        left, right = merge if isinstance(merge, list) else merge.split(" ")
        parents[N_BYTE_TOKENS + rank] = (vocab[left], vocab[right])
    return parents


def required_with(token: int, keep: set[int], parents: dict) -> set[int]:
    """The ids that must join ``keep`` for ``token`` to be reachable by BPE."""
    needed: set[int] = set()
    stack = [token]
    while stack:
        current = stack.pop()
        if current in keep or current in needed:
            continue
        needed.add(current)
        stack.extend(parents.get(current, ()))
    return needed


def select_vocab(
    counts: Counter,
    parents: dict,
    base_size: int,
    min_freq: int,
    target_rows: int | None,
    n_added: int,
) -> list[int]:
    keep = set(range(N_BYTE_TOKENS))
    for token in sorted(t for t, c in counts.items() if t < base_size and c >= min_freq):
        keep |= required_with(token, keep, parents)
    print(f"  closure of tokens with freq >= {min_freq}: {len(keep):,} rows")

    if target_rows is not None:
        budget = target_rows - n_added
        if budget < len(keep):
            raise SystemExit(
                f"--target-rows {target_rows:,} is below the {len(keep) + n_added:,} rows "
                f"needed at --min-freq {min_freq}"
            )
        # Low merge rank tracks general English frequency, so spend the headroom
        # from the bottom: these are common words the read-speech corpus missed.
        for token in range(base_size):
            if len(keep) >= budget:
                break
            if token in keep:
                continue
            needed = required_with(token, keep, parents)
            if len(keep) + len(needed) <= budget:
                keep |= needed
        print(f"  padded to target: {len(keep):,} rows")

    return sorted(keep)


def remap_tokenizer(source: dict, keep: list[int]) -> tuple[dict, dict[int, int]]:
    tokenizer_json = copy.deepcopy(source)
    model = tokenizer_json["model"]
    token_of = {i: t for t, i in model["vocab"].items()}

    new_id = {old: index for index, old in enumerate(keep)}
    model["vocab"] = {token_of[old]: index for old, index in new_id.items()}
    kept = set(keep)
    model["merges"] = [
        merge
        for rank, merge in enumerate(model["merges"])
        if N_BYTE_TOKENS + rank in kept
    ]

    added = sorted(tokenizer_json["added_tokens"], key=lambda entry: entry["id"])
    for offset, entry in enumerate(added):
        new_id[entry["id"]] = len(keep) + offset
        entry["id"] = len(keep) + offset
    tokenizer_json["added_tokens"] = added
    return tokenizer_json, new_id


def remap_config(config: dict, new_id: dict[int, int], vocab_size: int) -> dict:
    config = copy.deepcopy(config)
    config["llm_config"]["vocab_size"] = vocab_size
    for scope in (config, config["llm_config"]):
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if scope.get(key) is not None:
                scope[key] = new_id[scope[key]]
    return config


def verify(source: Tokenizer, pruned: Tokenizer, new_id: dict[int, int], samples: list[str]) -> None:
    """The pruned tokenizer must segment corpus text identically to the source."""
    before = source.encode_batch_fast(samples)
    after = pruned.encode_batch_fast(samples)
    for text, old, new in zip(samples, before, after):
        expected = [new_id[token] for token in old.ids]
        if expected != new.ids:
            raise SystemExit(f"tokenization changed on: {text[:80]!r}")
    print(f"  verified identical tokenization on {len(samples):,} transcripts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("models/p4+p3/student_distil"))
    parser.add_argument("--output", type=Path, default=Path("models/final"))
    parser.add_argument("--transcripts", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument(
        "--target-rows",
        type=int,
        default=None,
        help="pad the closure with the lowest-id unused tokens up to this many rows",
    )
    parser.add_argument("--verify-samples", type=int, default=20000)
    args = parser.parse_args()

    tokenizer_json = json.loads((args.source / "tokenizer.json").read_text())
    base_size = len(tokenizer_json["model"]["vocab"])
    n_added = len(tokenizer_json["added_tokens"])
    source_tokenizer = Tokenizer.from_file(str(args.source / "tokenizer.json"))

    print(f"counting {args.transcripts}")
    counts = count_tokens(args.transcripts, source_tokenizer)
    used = sum(1 for t in counts if t < base_size)
    print(f"  {sum(counts.values()):,} tokens, {used:,} of {base_size:,} rows used")

    parents = merge_parents(tokenizer_json["model"])
    keep = select_vocab(counts, parents, base_size, args.min_freq, args.target_rows, n_added)

    pruned_json, new_id = remap_tokenizer(tokenizer_json, keep)
    vocab_size = len(keep) + n_added

    samples = [text for text, _ in read_transcripts(args.transcripts)][: args.verify_samples]
    verify(source_tokenizer, Tokenizer.from_str(json.dumps(pruned_json)), new_id, samples)

    tensors = load_file(args.source / "model.safetensors")
    rows = keep + [old for old in sorted(new_id) if old >= base_size]
    before = tensors[EMBED_KEY].shape[0]
    tensors[EMBED_KEY] = tensors[EMBED_KEY][torch.tensor(rows)].contiguous()

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output / "model.safetensors"), metadata={"format": "pt"})
    (args.output / "tokenizer.json").write_text(json.dumps(pruned_json, ensure_ascii=False))
    config = json.loads((args.source / "config.json").read_text())
    (args.output / "config.json").write_text(
        json.dumps(remap_config(config, new_id, vocab_size), indent=2)
    )
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        if (args.source / name).exists():
            shutil.copy2(args.source / name, args.output / name)

    total = sum(t.numel() for t in tensors.values())
    saved = (before - vocab_size) * tensors[EMBED_KEY].shape[1]
    print(f"wrote {args.output}")
    print(f"  vocab {before:,} -> {vocab_size:,}")
    print(f"  params {(total + saved) / 1e6:.1f}M -> {total / 1e6:.1f}M (saved {saved / 1e6:.1f}M)")


if __name__ == "__main__":
    main()
