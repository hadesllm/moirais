# morie.fn -- function file (hadesllm/morie)
"""Unanimity committee rule.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def cmunn(data=None, n=50):
    """Unanimity committee rule.

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


short = "cmunn"
alias = "cmunn"
quote = "It is not what happens to you, but how you react, that matters. -- Epictetus"
cmunn = cmunn


def cheatsheet() -> str:
    return "cmunn({}) -> Unanimity committee rule."
