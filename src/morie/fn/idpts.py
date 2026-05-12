# morie.fn -- function file (hadesllm/morie)
"""Text scaling ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpts(data=None, n=50):
    """Text scaling ideal point.

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


short = "idpts"
alias = "idpts"
quote = "The spice must flow. -- Paul Atreides"
idpts = idpts


def cheatsheet() -> str:
    return "idpts({}) -> Text scaling ideal point."
