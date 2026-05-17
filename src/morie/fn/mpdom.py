# morie.fn -- function file (hadesllm/morie)
"""Dominant party model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpdom(data=None, n=50):
    """Dominant party model.

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


short = "mpdom"
alias = "mpdom"
quote = "Give me a place to stand and I will move the earth. -- Archimedes"
mpdom = mpdom


def cheatsheet() -> str:
    return "mpdom({}) -> Dominant party model."
