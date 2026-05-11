# morie.fn — function file (hadesllm/morie)
"""Lorenz curve data."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def lorenz_curve(
    values: np.ndarray | list[float],
) -> DescriptiveResult:
    """Compute Lorenz curve coordinates.

    Parameters
    ----------
    values : array-like
        Non-negative values.

    Returns
    -------
    DescriptiveResult
        extra contains 'proportions' (population %) and 'cumulative_share' (value %).
    """
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) < 2:
        raise ValueError("Need at least 2 values")
    n = len(v)
    cumsum = np.cumsum(v)
    total = cumsum[-1]
    proportions = np.arange(1, n + 1) / n
    cumulative_share = cumsum / total if total > 0 else np.zeros(n)
    proportions = np.concatenate([[0], proportions])
    cumulative_share = np.concatenate([[0], cumulative_share])
    return DescriptiveResult(
        name="lorenz_curve",
        value=float(n),
        extra={
            "proportions": proportions.tolist(),
            "cumulative_share": cumulative_share.tolist(),
            "n": n,
        },
    )


eqlrz = lorenz_curve


def cheatsheet() -> str:
    return "lorenz_curve({}) -> Lorenz curve data."
