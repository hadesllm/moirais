# morie.fn — function file (hadesllm/morie)
"""Learning rate finder from loss curve."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lr_finder(
    losses: list[float] | np.ndarray,
    lrs: list[float] | np.ndarray,
    smoothing: float = 0.05,
) -> DescriptiveResult:
    """Find optimal learning rate from a loss-vs-LR sweep.

    Returns the LR at which the smoothed loss has the steepest negative slope
    (point of maximum descent).

    :param losses: Loss values recorded during LR sweep.
    :param lrs: Corresponding learning rates.
    :param smoothing: Exponential smoothing factor.
    :return: DescriptiveResult with optimal LR.
    """
    losses = np.asarray(losses, dtype=np.float64)
    lrs = np.asarray(lrs, dtype=np.float64)
    if len(losses) != len(lrs):
        raise ValueError("losses and lrs must have the same length")
    if len(losses) < 3:
        raise ValueError("Need at least 3 data points")

    smoothed = np.empty_like(losses)
    smoothed[0] = losses[0]
    for i in range(1, len(losses)):
        smoothed[i] = smoothing * losses[i] + (1 - smoothing) * smoothed[i - 1]

    grad = np.gradient(smoothed, np.log(lrs + 1e-30))
    best_idx = int(np.argmin(grad))
    optimal_lr = float(lrs[best_idx])

    return DescriptiveResult(
        name="lr_finder",
        value=optimal_lr,
        extra={"best_idx": best_idx, "min_grad": float(grad[best_idx]), "smoothed_losses": smoothed},
    )


def cheatsheet() -> str:
    return "lr_finder(losses, lrs) -> optimal LR from loss curve sweep"


lrinf = lr_finder
