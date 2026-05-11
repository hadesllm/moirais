# morie.fn — function file (hadesllm/morie)
"""Party bloc positioning.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppblk(data=None, n=50):
    """Party bloc positioning.

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


short = "ppblk"
alias = "ppblk"
quote = "The spice must flow. -- Paul Atreides"
ppblk = ppblk


def cheatsheet() -> str:
    return "ppblk({}) -> Party bloc positioning."
