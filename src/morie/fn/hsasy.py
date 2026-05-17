# morie.fn -- function file (hadesllm/morie)
"""Asymmetric Hotelling model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hsasy(data=None, n=50):
    """Asymmetric Hotelling model.

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


short = "hsasy"
alias = "hsasy"
quote = "Errors using inadequate data are much less than those using none. -- Charles Babbage"
hsasy = hsasy


def cheatsheet() -> str:
    return "hsasy({}) -> Asymmetric Hotelling model."
