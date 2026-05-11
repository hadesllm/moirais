# morie.fn — function file (hadesllm/morie)
"""Party split model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpspl(data=None, n=50):
    """Party split model.

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


short = "mpspl"
alias = "mpspl"
quote = "The spice must flow. -- Paul Atreides"
mpspl = mpspl


def cheatsheet() -> str:
    return "mpspl({}) -> Party split model."
