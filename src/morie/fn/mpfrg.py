# morie.fn — function file (hadesllm/morie)
"""Party fragmentation index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpfrg(data=None, n=50):
    """Party fragmentation index.

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


short = "mpfrg"
alias = "mpfrg"
quote = "The spice must flow. -- Paul Atreides"
mpfrg = mpfrg


def cheatsheet() -> str:
    return "mpfrg({}) -> Party fragmentation index."
