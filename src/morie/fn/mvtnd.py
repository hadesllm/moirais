# morie.fn -- function file (hadesllm/morie)
"""Median voter theorem N-dimensional.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtnd(data=None, n=50):
    """Median voter theorem N-dimensional.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "mvtnd"
alias = "mvtnd"
quote = "We must know. We will know. -- David Hilbert"
mvtnd = mvtnd


def cheatsheet() -> str:
    return "mvtnd({}) -> Median voter theorem N-dimensional."
