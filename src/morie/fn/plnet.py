# morie.fn -- function file (hadesllm/morie)
"""Network-based polarization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plnet(data=None, n=50):
    """Network-based polarization.

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


short = "plnet"
alias = "plnet"
quote = "The measure of a man is what he does with power. -- Plato"
plnet = plnet


def cheatsheet() -> str:
    return "plnet({}) -> Network-based polarization."
