# morie.fn — function file (hadesllm/morie)
"""Weighted aggregation spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pawgt(data=None, positions=None, weights=None, n=50):
    """Weighted aggregation spatial.

    Parameters
    ----------
    data : array_like, optional
        Data values.
    positions : array_like, optional
        Position values (unused, kept for API compat).
    weights : array_like, optional
        Weights for aggregation.
    n : int
        Number of random samples if data is None.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(data)
    if weights is not None:
        weights = np.atleast_1d(weights)
        stat = float(np.average(data, weights=weights))
    else:
        stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "pawgt"
alias = "pawgt"
quote = "The spice must flow. -- Paul Atreides"
pawgt = pawgt


def cheatsheet() -> str:
    return "pawgt({}) -> Weighted aggregation spatial."
