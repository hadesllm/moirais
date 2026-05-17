# morie.fn -- function file (hadesllm/morie)
"""Nonparametric spatial probability.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def psnon(data=None, n=50):
    """Nonparametric spatial probability.

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


short = "psnon"
alias = "psnon"
quote = "Errors using inadequate data are much less than those using none. -- Charles Babbage"
psnon = psnon


def cheatsheet() -> str:
    return "psnon({}) -> Nonparametric spatial probability."
