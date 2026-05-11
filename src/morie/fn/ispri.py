# morie.fn — function file (hadesllm/morie)
"""Issue priority weighting.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ispri(data=None, n=50):
    """Issue priority weighting.

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


short = "ispri"
alias = "ispri"
quote = "The spice must flow. -- Paul Atreides"
ispri = ispri


def cheatsheet() -> str:
    return "ispri({}) -> Issue priority weighting."
