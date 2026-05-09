# moirais.fn — function file (hadesllm/moirais)
"""Separable preferences multidim.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mdsep(data=None, n=50):
    """Separable preferences multidim.

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


short = "mdsep"
alias = "mdsep"
quote = "The spice must flow. -- Paul Atreides"
mdsep = mdsep


def cheatsheet() -> str:
    return "mdsep({}) -> Separable preferences multidim."
