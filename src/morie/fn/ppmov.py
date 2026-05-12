# morie.fn -- function file (hadesllm/morie)
"""Party movement model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppmov(data=None, n=50):
    """Party movement model.

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


short = "ppmov"
alias = "ppmov"
quote = "The spice must flow. -- Paul Atreides"
ppmov = ppmov


def cheatsheet() -> str:
    return "ppmov({}) -> Party movement model."
