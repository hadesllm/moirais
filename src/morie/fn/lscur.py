# morie.fn -- function file (hadesllm/morie)
"""Loss curve analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def loss_curve_analysis(
    losses: list[float] | np.ndarray,
    window: int = 10,
) -> DescriptiveResult:
    """Analyze a training loss curve for convergence, plateau, or divergence.

    :param losses: Sequence of training losses (one per step/epoch).
    :param window: Smoothing window size.
    :return: DescriptiveResult with status ('converging', 'plateau', 'diverging').
    """
    losses = np.asarray(losses, dtype=np.float64)
    if len(losses) < 3:
        raise ValueError("Need at least 3 loss values")

    w = min(window, len(losses))
    smoothed = np.convolve(losses, np.ones(w) / w, mode="valid")

    if len(smoothed) < 2:
        status = "insufficient_data"
        slope = 0.0
    else:
        diffs = np.diff(smoothed)
        slope = float(np.mean(diffs))
        std_diff = float(np.std(diffs))

        if slope < -1e-6:
            status = "converging"
        elif abs(slope) < 1e-6 or std_diff < 1e-6:
            status = "plateau"
        else:
            status = "diverging"

    return DescriptiveResult(
        name="loss_curve_analysis",
        value=status,
        extra={
            "slope": float(slope),
            "final_loss": float(losses[-1]),
            "min_loss": float(np.min(losses)),
            "n_steps": len(losses),
            "smoothed": smoothed,
        },
    )


def cheatsheet() -> str:
    return "loss_curve_analysis(losses) -> detect plateau/divergence/convergence"


lscur = loss_curve_analysis
