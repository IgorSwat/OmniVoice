"""CSV-manifest dataset for calibration, repair, distillation and eval.

The training pipeline in ``omnivoice.training.builder`` reads WebDataset tar
shards. The pruning work only ever needs pre-tokenized codec ``.npy`` files, so
this reads the ``|``-delimited CSV manifests in ``data/`` directly and skips the
shard-conversion step entirely.

Manifest columns: ``name|transcription|speaker_id|language``, where ``name`` is
``<dirs>/<basename>`` and the codec lives at ``data/<dirs>/codecs/<basename>.npy``
with shape ``[num_codebooks, num_frames]``.
"""

import csv
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Codec frame rate of the Higgs audio tokenizer, used only for logging hours.
FRAME_RATE = 25


def codec_path(name: str, data_root: str = "data") -> str:
    dirs, base = name.rsplit("/", 1)
    return os.path.join(data_root, dirs, "codecs", f"{base}.npy")


def _npy_num_frames(path: str) -> int:
    # mmap reads only the header; never touches the payload.
    return int(np.load(path, mmap_mode="r").shape[1])


class CodecManifestDataset(Dataset):
    """Rows of a CSV manifest, yielding raw samples for ``OmniVoiceSampleProcessor``.

    Sample lengths are cached next to the manifest as ``<manifest>.lengths.npy``
    so that length-grouped batching does not re-stat 300k files on every run.
    """

    def __init__(
        self,
        manifest: str,
        data_root: str = "data",
        min_frames: int = 50,
        max_frames: int = 2000,
        limit: int = 0,
        cache_lengths: bool = True,
    ):
        self.data_root = data_root
        with open(manifest) as f:
            rows = list(csv.DictReader(f, delimiter="|"))
        if limit:
            rows = rows[:limit]

        lengths = self._load_or_build_lengths(manifest, rows, cache_lengths, limit)

        keep = [
            i
            for i, n in enumerate(lengths)
            if n >= min_frames and n <= max_frames and n > 0
        ]
        self.rows = [rows[i] for i in keep]
        self.lengths = [int(lengths[i]) for i in keep]
        self.dropped = len(rows) - len(self.rows)

    def _load_or_build_lengths(
        self, manifest: str, rows: List[dict], cache: bool, limit: int
    ) -> np.ndarray:
        cache_path = f"{manifest}.lengths.npy"
        if cache and not limit and os.path.exists(cache_path):
            cached = np.load(cache_path)
            if len(cached) == len(rows):
                return cached
        lengths = np.zeros(len(rows), dtype=np.int64)
        for i, r in enumerate(rows):
            p = codec_path(r["name"], self.data_root)
            lengths[i] = _npy_num_frames(p) if os.path.exists(p) else 0
        if cache and not limit:
            np.save(cache_path, lengths)
        return lengths

    @property
    def hours(self) -> float:
        return sum(self.lengths) / FRAME_RATE / 3600.0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.rows[i]
        tokens = np.load(codec_path(r["name"], self.data_root)).astype(np.int64)
        return {
            "audio_tokens": torch.from_numpy(tokens),
            "label": {
                "text": r["transcription"],
                "language_id": r.get("language", "en"),
            },
        }


class LengthGroupedBatchSampler(torch.utils.data.Sampler):
    """Groups similar-length samples so padding waste stays low.

    Batches are capped by a token budget (``max_len * batch_size <= batch_tokens``)
    rather than a fixed batch size, mirroring the repo's ``batch_tokens`` config.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_tokens: int,
        max_batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
        prefix_slack: int = 48,
    ):
        # Text/style prefix adds tokens the codec length does not account for.
        self.lengths = [n + prefix_slack for n in lengths]
        self.batch_tokens = batch_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._batches = self._build()

    def _build(self) -> List[List[int]]:
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches, cur, cur_max = [], [], 0
        for i in order:
            nxt_max = max(cur_max, self.lengths[i])
            if cur and (
                nxt_max * (len(cur) + 1) > self.batch_tokens
                or len(cur) + 1 > self.max_batch_size
            ):
                batches.append(cur)
                cur, cur_max = [i], self.lengths[i]
            else:
                cur.append(i)
                cur_max = nxt_max
        if cur:
            batches.append(cur)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches)


def build_processor(model_config, tokenizer, deterministic: bool = False):
    """``OmniVoiceSampleProcessor`` at the checkpoint's training defaults.

    ``deterministic`` is unused by the processor itself (it draws from the global
    ``random`` module); seed via :func:`seed_everything` instead.
    """
    from omnivoice.data.processor import OmniVoiceSampleProcessor

    return OmniVoiceSampleProcessor(
        text_tokenizer=tokenizer,
        num_channels=model_config.num_audio_codebook,
        audio_mask_id=model_config.audio_mask_id,
        prompt_ratio_range=(0.0, 0.3),
        mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=0.1,
        language_ratio=0.8,
        use_pinyin_ratio=0.0,
        instruct_ratio=0.0,
        only_instruct_ratio=0.0,
    )


def build_dataloader(
    dataset: CodecManifestDataset,
    processor,
    batch_tokens: int,
    max_batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader emitting the 4D-bidirectional-mask batches the model expects.

    ``PaddingDataCollator`` builds ``attention_mask`` as ``[B, 1, L, L]`` where
    every query attends to every non-padding key. Passing a 4D mask stops
    HuggingFace from adding a causal mask on top, which would be wrong for this
    masked-diffusion model.
    """
    from omnivoice.data.collator import PaddingDataCollator

    collator = PaddingDataCollator(processor, batch_tokens)

    def collate(samples: List[Dict[str, Any]]):
        return collator([processor(s) for s in samples])

    sampler = LengthGroupedBatchSampler(
        dataset.lengths,
        batch_tokens=batch_tokens,
        max_batch_size=max_batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers else None,
    )
    loader.batch_sampler_ref = sampler
    return loader


def _seed_worker(worker_id: int) -> None:
    """Give each dataloader worker its own ``random``/``numpy`` stream.

    ``OmniVoiceSampleProcessor`` draws ``mask_ratio`` and ``prompt_ratio`` from
    the stdlib ``random`` module. PyTorch reseeds only ``torch`` per worker, and
    under the ``fork`` start method every worker inherits the parent's ``random``
    state — so worker 0 and worker 1 would draw the *same* sequence of noise
    levels, collapsing the diversity of the diffusion time axis that calibration
    and training both depend on.
    """
    base = torch.initial_seed() % (2**31)
    random.seed(base + worker_id)
    np.random.seed((base + worker_id) % (2**31))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
