"""Closed-form least-squares repair of the write-side projections.

Implements section 5 of ``knowledge/width_pruning.md``.

After truncation the outputs are wrong by the projection residual. Rather than
fine-tune it away, refit each write-side projection to best reproduce the
teacher's output given the *student's* inputs::

    min_W  || Qk^T (W_teacher h_teacher)  -  W h_student ||^2  + lambda ||W||^2

This is solved by normal equations — no gradients, minutes — and it is what makes
width pruning cheap to evaluate: a candidate can be cut, repaired and scored in
seconds. Depth pruning has no equivalent.

Only ``XtX`` and ``XtY`` are accumulated; raw activations are never stored.

A note on the codebook weighting mentioned in the knowledge doc: for a closed-form
refit of ``audio_heads`` the weights are provably a no-op. The objective
``sum_c w_c || y_c - W_c h ||^2`` decouples over codebooks because each codebook
owns a disjoint block of output rows and the design matrix ``XtX`` is shared, so a
positive ``w_c`` rescales each subproblem without moving its argmin. The codebook
weights do matter in the distillation loss (see ``distill.py``), where the
objective is genuinely joint.
"""

from typing import Dict, List, Optional, Tuple

import torch

from .manifest import to_device


class _Accumulator:
    """Accumulates ``X^T X``, ``X^T Y`` and ``tr(Y^T Y)`` in float64 on CPU."""

    def __init__(self, in_dim: int, out_dim: int):
        self.A = torch.zeros(in_dim, in_dim, dtype=torch.float64)
        self.B = torch.zeros(in_dim, out_dim, dtype=torch.float64)
        self.yy = 0.0
        self.n = 0

    def add(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        X = X.float()
        Y = Y.float()
        self.A += (X.T @ X).cpu().double()
        self.B += (X.T @ Y).cpu().double()
        self.yy += float((Y * Y).sum())
        self.n += X.shape[0]

    @property
    def conditioning(self) -> float:
        """Positions accumulated per input dimension.

        Below 1.0 the normal equations are underdetermined and the refit
        interpolates the calibration set exactly -- a reported error of 0.0000 is
        overfitting, not success. ``down_proj`` has 3072 inputs, so a useful
        repair needs tens of thousands of positions; ~10x the input dimension is
        a reasonable floor.
        """
        return self.n / max(self.A.shape[0], 1)

    def solve(self, ridge_rel: float = 1e-4) -> torch.Tensor:
        """Return the refit weight in ``nn.Linear`` layout ``[out, in]``."""
        n = self.A.shape[0]
        lam = ridge_rel * float(torch.diagonal(self.A).sum()) / max(n, 1)
        A = self.A + lam * torch.eye(n, dtype=torch.float64)
        W = torch.linalg.solve(A, self.B)  # [in, out]
        return W.T.contiguous()

    def relative_error(self, W: torch.Tensor) -> float:
        """``||Y - X W^T||^2 / ||Y||^2`` evaluated from the accumulated moments."""
        Wt = W.T.double()  # [in, out]
        resid = (
            self.yy
            - 2.0 * float((Wt * self.B).sum())
            + float(torch.diagonal(Wt.T @ self.A @ Wt).sum())
        )
        return resid / max(self.yy, 1e-30)


def _write_targets(model) -> List[Tuple[str, torch.nn.Module]]:
    out = []
    for i, layer in enumerate(model.llm.layers):
        out.append((f"layers.{i}.self_attn.o_proj", layer.self_attn.o_proj))
        out.append((f"layers.{i}.mlp.down_proj", layer.mlp.down_proj))
    return out


def _valid_mask(batch) -> torch.Tensor:
    """Non-padding positions. ``attention_mask[b, 0, i, j] == valid[b, j]``."""
    return batch["attention_mask"][:, 0, 0, :]


def _position_mask(batch, positions: str) -> torch.Tensor:
    valid = _valid_mask(batch)
    if positions == "audio":
        return valid & batch["audio_mask"]
    return valid


@torch.no_grad()
def _collect(
    teacher,
    student,
    dataloader,
    device,
    Qk: Optional[torch.Tensor],
    modules: List[Tuple[str, torch.nn.Module, torch.nn.Module]],
    positions: str,
    max_batches: int,
    run_head: bool = False,
) -> Dict[str, _Accumulator]:
    """One paired pass, hooking student inputs and teacher outputs.

    ``modules`` is a list of ``(name, teacher_module, student_module)``. Teacher
    outputs are projected by ``Qk`` (write-side rotation) before being used as
    targets; pass ``Qk=None`` when the output dimension is unchanged, as for
    ``audio_heads``.
    """
    from .calibration import _hidden_states

    cache: Dict[str, dict] = {name: {} for name, _, _ in modules}
    handles = []

    def t_hook(name):
        def fn(_mod, _inp, out):
            cache[name]["y"] = out.detach()

        return fn

    def s_hook(name):
        def fn(_mod, inp, _out):
            cache[name]["x"] = inp[0].detach()

        return fn

    for name, tm, sm in modules:
        handles.append(tm.register_forward_hook(t_hook(name)))
        handles.append(sm.register_forward_hook(s_hook(name)))

    accs: Dict[str, _Accumulator] = {}
    # Project teacher outputs on-device in float32: MPS has no float64, and the
    # accumulator casts to float64 only after the reduction (see _Accumulator.add).
    Qk_dev = None if Qk is None else Qk.float().to(device)
    try:
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            batch = to_device(batch, device)
            _, t_last = _hidden_states(teacher, batch, output_hidden_states=False)
            _, s_last = _hidden_states(student, batch, output_hidden_states=False)
            if run_head:
                # _hidden_states runs only the backbone; audio_heads sits outside
                # it, so its forward hooks never fire unless it is called here.
                teacher.audio_heads(t_last)
                student.audio_heads(s_last)
            mask = _position_mask(batch, positions)

            for name, _, sm in modules:
                x = cache[name]["x"][mask]
                y = cache[name]["y"][mask]
                if Qk_dev is not None:
                    y = y.float() @ Qk_dev
                if name not in accs:
                    accs[name] = _Accumulator(x.shape[-1], y.shape[-1])
                accs[name].add(x, y)
                cache[name].clear()
    finally:
        for h in handles:
            h.remove()
    return accs


def _module_by_name(model, name: str):
    mod = model.llm
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


def repair(
    teacher,
    student,
    dataloader,
    device,
    Q: torch.Tensor,
    k: int,
    mode: str = "sequential",
    positions: str = "all",
    ridge_rel: float = 1e-4,
    max_batches: int = 0,
    repair_audio_heads: bool = True,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Refit ``o_proj``/``down_proj`` (and ``audio_heads``) in closed form.

    ``teacher`` is the model as it stood *before this rung's cut* — repair is a
    local reconstruction problem, not a distillation one. The original 1024-wide
    teacher is used for the KD recovery instead.

    ``mode``:

    * ``sequential`` (default) — one pass per layer, so that when layer ``l`` is
      refit, layers below it have already been repaired and the student inputs
      are the ones it will actually see. Costs ``num_layers + 1`` passes over the
      calibration set and holds one layer's normal equations at a time (~130 MB).
      Within a layer, ``o_proj`` and ``down_proj`` are refit together, so
      ``down_proj`` sees pre-repair ``o_proj`` inputs — a second-order effect.
    * ``joint`` — a single pass refitting every layer against the unrepaired
      student. Faster, but ignores that repairing layer ``l`` changes the inputs
      of layer ``l+1``. Holds all normal equations at once (~4 GB at d=1024).

    Returns a report mapping module name to before/after relative error.
    """
    Qk = Q[:, :k].detach().cpu().double()
    teacher.eval()
    student.eval()
    report: Dict[str, dict] = {}

    def _apply(name: str, acc: _Accumulator) -> None:
        sm = _module_by_name(student, name)
        before = acc.relative_error(sm.weight.data.detach().cpu().double())
        W = acc.solve(ridge_rel)
        after = acc.relative_error(W)
        from .surgery import _set_linear

        _set_linear(sm, W)
        report[name] = {
            "before": before,
            "after": after,
            "positions": acc.n,
            "positions_per_input_dim": acc.conditioning,
        }
        if verbose:
            warn = "  <-- UNDERDETERMINED" if acc.conditioning < 10 else ""
            print(
                f"    {name:<34} rel.err {before:.4f} -> {after:.4f}  "
                f"({acc.n} positions, {acc.conditioning:.1f}x in_dim){warn}",
                flush=True,
            )

    num_layers = len(student.llm.layers)

    if mode == "joint":
        mods = [
            (name, _module_by_name(teacher, name), _module_by_name(student, name))
            for name, _ in _write_targets(student)
        ]
        accs = _collect(
            teacher, student, dataloader, device, Qk, mods, positions, max_batches
        )
        for name, _, _ in mods:
            _apply(name, accs[name])
    elif mode == "sequential":
        for i in range(num_layers):
            if verbose:
                print(f"  repair pass {i + 1}/{num_layers + 1} (layer {i})", flush=True)
            names = [f"layers.{i}.self_attn.o_proj", f"layers.{i}.mlp.down_proj"]
            mods = [
                (n, _module_by_name(teacher, n), _module_by_name(student, n))
                for n in names
            ]
            accs = _collect(
                teacher, student, dataloader, device, Qk, mods, positions, max_batches
            )
            for n in names:
                _apply(n, accs[n])
    else:
        raise ValueError(f"unknown repair mode: {mode}")

    if repair_audio_heads:
        if verbose:
            print(f"  repair pass {num_layers + 1}/{num_layers + 1} (audio_heads)")
        mods = [("audio_heads", teacher.audio_heads, student.audio_heads)]
        # audio_heads is a READ of the stream: its output dimension (8 * 1025) is
        # unchanged by width pruning, so the teacher target needs no projection.
        accs = _collect(
            teacher,
            student,
            dataloader,
            device,
            None,
            mods,
            positions,
            max_batches,
            run_head=True,
        )
        acc = accs["audio_heads"]
        before = acc.relative_error(student.audio_heads.weight.data.detach().cpu().double())
        W = acc.solve(ridge_rel)
        after = acc.relative_error(W)
        from .surgery import _set_linear

        _set_linear(student.audio_heads, W)
        report["audio_heads"] = {
            "before": before,
            "after": after,
            "positions": acc.n,
            "positions_per_input_dim": acc.conditioning,
        }
        if verbose:
            warn = "  <-- UNDERDETERMINED" if acc.conditioning < 10 else ""
            print(
                f"    {'audio_heads':<34} rel.err {before:.4f} -> {after:.4f}  "
                f"({acc.n} positions, {acc.conditioning:.1f}x in_dim){warn}"
            )

    return report
