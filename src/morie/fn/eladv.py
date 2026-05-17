# morie.fn -- function file (hadesllm/morie)
"""Incumbency advantage model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def eladv(data=None, n=50):
    """Incumbency advantage model.

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


short = "eladv"
alias = "eladv"
quote = "Give me a place to stand and I will move the earth. -- Archimedes"
eladv = eladv


def cheatsheet() -> str:
    return "eladv({}) -> Incumbency advantage model."
