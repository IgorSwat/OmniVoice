"""KD recovery training for a freshly pruned student.

Section 11 of ``knowledge/width_pruning.md`` is the framing: this is not one-shot
surgery. Each cut is followed by a short knowledge-distillation recovery, and the
target is quality *after* recovery. Post-surgery loss is a weak predictor of
final quality — a proxy whose correlation must be verified, not assumed.

The recovery always distills from the ORIGINAL teacher, never from the previous
rung's student, so that errors do not compound down the ladder.

Loss terms:

* ``kd`` — KL(teacher || student) over each codebook's distribution at masked
  positions, weighted by ``audio_codebook_weights = [8,8,6,6,4,4,2,2]``
  normalized as in ``OmniVoice.__init__``. Codebook 0 carries 4x the weight of
  codebook 7, and unlike the closed-form repair this objective is genuinely
  joint, so the weights actually bite here.
* ``ce`` — the model's own cross-entropy against ground-truth codes.
* ``hidden`` — optional per-boundary match of the student's residual stream to
  the teacher's, projected into the student basis. Width pruning is unusual in
  giving an *exact* linear map between teacher and student streams (the
  cumulative rotation), so this is well defined rather than heuristic. It is
  exact at initialization and degrades as training drifts the student away from
  the rotated basis, so treat it as a warm-start regularizer. Per-boundary
  normalization is mandatory: residual energy grows ~75,000x from boundary 0 to
  27, so an unnormalized sum would be decided entirely by the deepest layers.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .calibration import _audio_logits, _hidden_states
from .manifest import to_device


@dataclass
class DistillConfig:
    steps: int = 2000
    lr: float = 1e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    grad_accum: int = 1
    kd_weight: float = 1.0
    ce_weight: float = 0.0
    hidden_weight: float = 0.0
    kd_reverse: bool = False
    temperature: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 0
    amp_dtype: str = "bf16"


def _lr_at(step: int, cfg: DistillConfig) -> float:
    warmup = max(int(cfg.warmup_ratio * cfg.steps), 1)
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    p = (step - warmup) / max(cfg.steps - warmup, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos)


def _codebook_weights(model, device) -> torch.Tensor:
    return torch.tensor(model.normalized_audio_codebook_weights, device=device)


LOG_FLOOR = -20.7   # log(1e-9)


def _weighted_ce(logits: torch.Tensor, labels: torch.Tensor, w: torch.Tensor):
    """Reproduces ``OmniVoice.forward``'s loss: per-codebook mean, then weighted sum."""
    per_token = F.cross_entropy(
        logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=-100
    )
    valid = (labels != -100).float()
    per_cb = (per_token * valid).sum(dim=(0, 2)) / valid.sum(dim=(0, 2)).clamp(min=1.0)
    return (per_cb * w).sum()


def _weighted_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    w: torch.Tensor,
    temperature: float = 1.0,
    reverse: bool = False,
):
    """Codebook-weighted KL on loss-bearing positions.

    ``reverse=False`` is KL(teacher || student); ``True`` is KL(student || teacher).
    """
    mask = (labels != -100).float()  # [B, C, S]
    t = temperature
    log_p_s = F.log_softmax(student_logits / t, dim=-1)
    log_p_t = F.log_softmax(teacher_logits / t, dim=-1)
    if reverse:
        # KL(student || teacher): mode-seeking. Weighted by the student's own
        # probability, so it punishes mass the student places where the teacher
        # would not, rather than mass it fails to cover. For a capacity-limited
        # student that is usually the better compromise -- forward KL prefers a
        # spread-out fit that samples from regions the teacher rates ~zero, and
        # in an iterative sampler one such commit conditions every later step.
        # LOG_FLOOR keeps it finite where the teacher's probability underflows.
        kl = (log_p_s.exp() * (log_p_s - log_p_t.clamp_min(LOG_FLOOR))).sum(dim=-1)
    else:
        p_t = log_p_t.exp()
        kl = torch.where(p_t > 0, p_t * (log_p_t - log_p_s),
                         torch.zeros_like(p_t)).sum(dim=-1)  # [B, C, S]
    per_cb = (kl * mask).sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp(min=1.0)
    return (per_cb * w).sum() * (t * t)


def _hidden_match(student_hs, teacher_hs, projection: torch.Tensor, valid: torch.Tensor):
    """Per-boundary normalized MSE between student and rotated teacher streams."""
    total = 0.0
    # float32 on-device: the projection is stored in float64 on CPU, which MPS
    # cannot hold and which would needlessly upcast the matmul on CUDA.
    P = projection.float().to(student_hs[0].device)
    for hs, ht in zip(student_hs, teacher_hs):
        s = hs[valid].float()
        t = ht[valid].float() @ P
        denom = t.pow(2).sum().clamp(min=1e-6)
        total = total + (s - t).pow(2).sum() / denom
    return total / max(len(student_hs), 1)


EVAL_SEED = 1234


@torch.no_grad()
def evaluate_loss(
    model, dataloader, device, max_batches: int = 0, seed: int = EVAL_SEED
) -> float:
    """Mean codebook-weighted CE on the dev set — the gate metric.

    The RNG is pinned for the duration of the pass. ``OmniVoiceSampleProcessor``
    redraws ``mask_ratio ~ U(0, 1)``, ``prompt_ratio`` and the per-token mask on
    every ``__call__``, so without this the teacher's baseline (measured once at
    startup) and the student's evals (at each checkpoint) land on *different*
    noise levels. Measured across runs that moves the dev loss by up to 0.13
    nats, several times the ~0.047 gap a 1% gate is trying to resolve, so the
    gate would otherwise be comparing numbers below its own noise floor.

    Both generators are restored afterwards: reseeding them permanently would
    make every training batch after the first eval replay the same masks.

    Requires ``num_workers=0``; masks drawn in worker processes are outside
    this seeding.
    """
    py_state = random.getstate()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)

    was_training = model.training
    model.eval()
    w = _codebook_weights(model, device)
    total, count = 0.0, 0
    try:
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            batch = to_device(batch, device)
            _, last = _hidden_states(model, batch, output_hidden_states=False)
            loss = _weighted_ce(_audio_logits(model, last), batch["labels"], w)
            total += float(loss)
            count += 1
    finally:
        random.setstate(py_state)
        torch.set_rng_state(torch_state)
        if was_training:
            model.train()
    return total / max(count, 1)


def distill(
    student,
    teacher,
    train_loader,
    dev_loader,
    device,
    cfg: DistillConfig,
    projection: Optional[torch.Tensor] = None,
    out_dir: Optional[str] = None,
    verbose: bool = True,
    teacher_logits_fn=None,
) -> Dict[str, float]:
    """Run KD recovery. Returns a summary with the final dev loss.

    ``projection`` is the cumulative ``[d_teacher, d_student]`` map from the
    original teacher's residual basis to the student's, required only when
    ``cfg.hidden_weight > 0``.

    ``teacher_logits_fn(teacher, batch) -> [B, C, S, V]`` overrides how the KD
    target is produced. It exists so the teacher can run classifier-free guidance
    (two branches, mixed) while the student runs a single conditional pass -- the
    setup needed to prune a guidance-distilled student against an ORIGINAL,
    un-distilled teacher. Incompatible with ``hidden_weight > 0``, which needs the
    teacher's per-boundary hidden states.
    """
    if teacher_logits_fn is not None and cfg.hidden_weight > 0:
        raise ValueError("teacher_logits_fn cannot be combined with hidden_weight > 0")
    use_hidden = cfg.hidden_weight > 0.0
    if use_hidden and projection is None:
        raise ValueError("hidden_weight > 0 requires a cumulative projection")

    # With no KD and no hidden matching the teacher contributes nothing to the
    # loss, so running it every step is pure waste -- and with a CFG teacher it is
    # two wasted forwards. ``teacher`` may then be None.
    need_teacher = cfg.kd_weight > 0.0 or use_hidden
    if not need_teacher and verbose:
        print("  teacher forward SKIPPED (kd_weight=0, hidden_weight=0)")
    if need_teacher and teacher is None:
        raise ValueError("a teacher is required unless kd_weight and hidden_weight "
                         "are both 0")

    amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype is torch.float16)

    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    student.train()

    decay, no_decay = [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )

    w = _codebook_weights(student, device)
    step, micro = 0, 0
    running: Dict[str, float] = {"loss": 0.0, "kd": 0.0, "ce": 0.0, "hidden": 0.0}
    t_start = time.time()
    history = []
    epoch = 0
    it = iter(train_loader)

    while step < cfg.steps:
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            sampler = getattr(train_loader, "batch_sampler_ref", None)
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(train_loader)
            batch = next(it)

        batch = to_device(batch, device)
        valid = batch["attention_mask"][:, 0, 0, :].bool()

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            with torch.no_grad():
                if not need_teacher:
                    t_hs = t_logits = None
                elif teacher_logits_fn is not None:
                    t_hs, t_logits = None, teacher_logits_fn(teacher, batch)
                else:
                    t_hs, t_last = _hidden_states(
                        teacher, batch, output_hidden_states=use_hidden
                    )
                    t_logits = _audio_logits(teacher, t_last)
            s_hs, s_last = _hidden_states(
                student, batch, output_hidden_states=use_hidden
            )
            s_logits = _audio_logits(student, s_last)

            kd = (_weighted_kd(s_logits, t_logits, batch["labels"], w,
                               cfg.temperature, reverse=cfg.kd_reverse)
                  if t_logits is not None
                  else torch.zeros((), device=device))
            ce = (
                _weighted_ce(s_logits, batch["labels"], w)
                if cfg.ce_weight > 0
                else torch.zeros((), device=device)
            )
            hid = (
                _hidden_match(s_hs, t_hs, projection, valid)
                if use_hidden
                else torch.zeros((), device=device)
            )
            loss = cfg.kd_weight * kd + cfg.ce_weight * ce + cfg.hidden_weight * hid

        scaler.scale(loss / cfg.grad_accum).backward()
        # .detach() before the scalar read: keeps the autograd graph from being
        # held alive by these locals across the grad-accumulation `continue`.
        running["loss"] += float(loss.detach())
        running["kd"] += float(kd.detach())
        running["ce"] += float(ce.detach())
        running["hidden"] += float(hid.detach())
        micro += 1

        if micro % cfg.grad_accum != 0:
            continue

        lr = _lr_at(step, cfg)
        for g in optim.param_groups:
            g["lr"] = lr
        if cfg.max_grad_norm > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        step += 1

        if verbose and cfg.log_every and step % cfg.log_every == 0:
            k = cfg.log_every * cfg.grad_accum
            elapsed = time.time() - t_start
            print(
                f"    step {step}/{cfg.steps}  loss {running['loss'] / k:.4f}  "
                f"kd {running['kd'] / k:.4f}  ce {running['ce'] / k:.4f}  "
                f"hid {running['hidden'] / k:.4f}  lr {lr:.2e}  "
                f"{step / max(elapsed, 1e-6):.2f} step/s",
                flush=True,
            )
            running = {key: 0.0 for key in running}

        if cfg.eval_every and step % cfg.eval_every == 0 and dev_loader is not None:
            dl = evaluate_loss(student, dev_loader, device)
            history.append({"step": step, "dev_loss": dl})
            if verbose:
                print(f"    step {step}: dev loss {dl:.4f}", flush=True)
            student.train()

        if out_dir and cfg.save_every and step % cfg.save_every == 0:
            from .surgery import save_pruned

            save_pruned(student, f"{out_dir}/step_{step}")

    final = (
        evaluate_loss(student, dev_loader, device) if dev_loader is not None else None
    )
    return {
        "steps": step,
        "dev_loss": final,
        "history": history,
        "wall_seconds": time.time() - t_start,
    }
