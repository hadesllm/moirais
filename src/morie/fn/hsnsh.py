# morie.fn -- function file (hadesllm/morie)
"""Nash equilibrium in Hotelling.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hsnsh(data=None, n=50):
    """Nash equilibrium in Hotelling.

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


short = "hsnsh"
alias = "hsnsh"
quote = "The spice must flow. -- Paul Atreides"
hsnsh = hsnsh


def cheatsheet() -> str:
    return "hsnsh({}) -> Nash equilibrium in Hotelling."
