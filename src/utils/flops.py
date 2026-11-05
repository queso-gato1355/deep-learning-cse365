"""Parameter and FLOP counting with PyTorch's built-in ``FlopCounterMode`` (no extra packages)."""

from __future__ import annotations

import logging

import torch

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)  # silence "triton"
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402


def count_params(model: torch.nn.Module) -> float:
    """Return the number of parameters in millions.

    Args:
        model: Any PyTorch module.

    Returns:
        Parameters (M).
    """
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gflops(model: torch.nn.Module, hw: tuple[int, int]) -> float:
    """Count forward-pass GFLOPs for one image (multiply and add each count as one FLOP).

    Args:
        model: Model in eval mode.
        hw: Input ``(height, width)``.

    Returns:
        GFLOPs.
    """
    device = next(model.parameters()).device
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.zeros(1, 3, *hw, device=device))
    return fc.get_total_flops() / 1e9
