# morie.fn -- function file (hadesllm/morie)
"""MAP test dimensionality.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtmap(data=None, n=50):
    """MAP test dimensionality.

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


short = "dtmap"
alias = "dtmap"
quote = "The spice must flow. -- Paul Atreides"
dtmap = dtmap


def cheatsheet() -> str:
    return "dtmap({}) -> MAP test dimensionality."
