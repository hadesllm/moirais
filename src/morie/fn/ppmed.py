# morie.fn — function file (hadesllm/morie)
"""Party median positioning.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppmed(data=None, n=50):
    """Party median positioning.

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


short = "ppmed"
alias = "ppmed"
quote = "The spice must flow. -- Paul Atreides"
ppmed = ppmed


def cheatsheet() -> str:
    return "ppmed({}) -> Party median positioning."
