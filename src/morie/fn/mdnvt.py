# morie.fn -- function file (hadesllm/morie)
"""Median voter theorem computation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def median_voter(positions, *, weights=None) -> DescriptiveResult:
    """Compute the median voter position (Black 1958).

    In single-peaked preferences over 1D, the median voter's ideal point
    is a Condorcet winner. Extends to weighted voters.

    :param positions: 1D array of voter ideal points.
    :param weights: Optional voter weights (e.g. population).
    :return: DescriptiveResult with median position.

    References
    ----------
    Armstrong (2014), Ch 1. Black (1958).

    .. epigraph:: Logic is the foundation of all certain knowledge. -- Leonhard Euler
    """
    x = np.asarray(positions, dtype=float).ravel()
    if len(x) == 0:
        raise ValueError("positions must be non-empty.")

    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if len(w) != len(x):
            raise ValueError("weights must match positions length.")
        order = np.argsort(x)
        x_sorted = x[order]
        w_sorted = w[order]
        cum = np.cumsum(w_sorted)
        half = cum[-1] / 2.0
        idx = np.searchsorted(cum, half)
        median_pos = float(x_sorted[min(idx, len(x_sorted) - 1)])
    else:
        median_pos = float(np.median(x))

    return DescriptiveResult(
        name="median_voter",
        value=median_pos,
        extra={"n_voters": len(x), "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0},
    )


mdnvt = median_voter


def cheatsheet() -> str:
    return "median_voter({}) -> Median voter theorem computation."
