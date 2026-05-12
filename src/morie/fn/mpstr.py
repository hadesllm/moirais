# morie.fn -- function file (hadesllm/morie)
"""Strategic entry multi-party.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpstr(data=None, n=50):
    """Strategic entry multi-party.

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


short = "mpstr"
alias = "mpstr"
quote = "The spice must flow. -- Paul Atreides"
mpstr = mpstr


def cheatsheet() -> str:
    return "mpstr({}) -> Strategic entry multi-party."
