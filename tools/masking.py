from typing import Literal

import numpy as np
import torch
from scipy.optimize import brentq

from tools.config import (
    KAPPA_0,
    KAPPA_NOISE,
    NO_CODEBOOKS,
    PHI_0_BOUNDS,
    PHI_1_BOUNDS,
    T_SHIFT,
    U_MAX,
)


def mask(
    L: int,
    batch_size: int = 1,
    mode: Literal["full", "uniform", "adaptive", "adaptive-remask"] = "full",
) -> torch.Tensor:
    """
    Draws a boolean mask of shape (batch_size, L, NO_CODEBOOKS).
    True marks a masked slot.

    Modes:
      - 'full': masks everything.
      - 'uniform': masks a single fraction R ~ U(0, 1) of every codebook.
      - 'adaptive': masks each codebook at the rate an inference step of progress
        u ~ U(0, 1) leaves behind, coarse codebooks resolving before deep ones.
      - 'adaptive-remask': same, with the two shallowest codebooks held back as remasking holds them back.
    """
    if mode == "full":
        return torch.ones(batch_size, L, NO_CODEBOOKS, dtype=torch.bool)
    if mode == "uniform":
        rates = torch.rand(batch_size, 1).expand(-1, NO_CODEBOOKS)
    elif mode in ("adaptive", "adaptive-remask"):
        rates = _adaptive_rates(batch_size, remask=mode == "adaptive-remask")
    else:
        raise ValueError(f"unknown mode: {mode}")

    # Bernoulli 
    return torch.rand(batch_size, L, NO_CODEBOOKS) < rates[:, None, :]


def _adaptive_rates(batch_size: int, remask: bool) -> torch.Tensor:
    """
    Returns per-codebook masking rates, shape (batch_size, NO_CODEBOOKS).
    """

    kappa = KAPPA_0 + torch.rand(batch_size, 1) * KAPPA_NOISE
    rho = torch.exp(-kappa * torch.arange(NO_CODEBOOKS))

    # adaptive-remask mode only
    if remask:
        rho[:, 0] *= torch.empty(batch_size).uniform_(*PHI_0_BOUNDS)
        rho[:, 1] *= torch.empty(batch_size).uniform_(*PHI_1_BOUNDS)

    u = torch.rand(batch_size) * U_MAX  # Works as drawing from U(0, U_MAX)
    R = (1 - u) / (1 - (1 - T_SHIFT) * u)

    # Theta is the race threshold making the eight rates average to R
    theta = [
        brentq(lambda t: np.exp(-t * rho_b).mean() - R_b, 0.0, 1e12)
        for rho_b, R_b in zip(rho.numpy(), R.numpy())
    ]

    return torch.exp(-torch.tensor(theta, dtype=rho.dtype)[:, None] * rho)
