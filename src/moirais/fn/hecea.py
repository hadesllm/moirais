# moirais.fn — function file (hadesllm/moirais)
"""Cost-effectiveness plane quadrant probabilities."""

import numpy as np

from ._containers import DescriptiveResult


def cost_effectiveness_plane(
    cost_diffs: list | np.ndarray,
    effect_diffs: list | np.ndarray,
) -> DescriptiveResult:
    """Compute CE plane quadrant probabilities from bootstrap samples.

    Parameters
    ----------
    cost_diffs : array-like
        Bootstrapped incremental cost differences.
    effect_diffs : array-like
        Bootstrapped incremental effect differences.

    Returns
    -------
    DescriptiveResult
    """
    c = np.asarray(cost_diffs, dtype=float)
    e = np.asarray(effect_diffs, dtype=float)
    if len(c) != len(e):
        raise ValueError("cost_diffs and effect_diffs must match")
    n = len(c)

    quadrants = {
        "NE": float(np.sum((c > 0) & (e > 0)) / n * 100),
        "NW": float(np.sum((c > 0) & (e <= 0)) / n * 100),
        "SE": float(np.sum((c <= 0) & (e > 0)) / n * 100),
        "SW": float(np.sum((c <= 0) & (e <= 0)) / n * 100),
    }

    return DescriptiveResult(
        name="ce_plane",
        value=quadrants,
        extra={"n_simulations": n, "mean_delta_c": float(np.mean(c)), "mean_delta_e": float(np.mean(e))},
    )


hecea = cost_effectiveness_plane


def cheatsheet() -> str:
    return "cost_effectiveness_plane({}) -> Cost-effectiveness plane quadrant probabilities."
