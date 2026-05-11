# morie.fn — function file (hadesllm/morie)
"""Committee ideal point estimation.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpcm(data=None, n=50):
    """Committee ideal point estimation.

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


short = "idpcm"
alias = "idpcm"
quote = "The spice must flow. -- Paul Atreides"
idpcm = idpcm


def cheatsheet() -> str:
    return "idpcm({}) -> Committee ideal point estimation."
