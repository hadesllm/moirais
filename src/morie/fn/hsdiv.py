# morie.fn -- function file (hadesllm/morie)
"""Divergence in Hotelling model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hsdiv(data=None, n=50):
    """Divergence in Hotelling model.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(np.asarray(data, dtype=float))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "hsdiv"
alias = "hsdiv"
quote = "You have power over your mind, not outside events. -- Marcus Aurelius"
hsdiv = hsdiv


def cheatsheet() -> str:
    return "hsdiv({}) -> Divergence in Hotelling model."
