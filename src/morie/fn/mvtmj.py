# morie.fn -- function file (hadesllm/morie)
"""Majority rule median voter.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtmj(data=None, n=50):
    """Majority rule median voter.

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


short = "mvtmj"
alias = "mvtmj"
quote = "The spice must flow. -- Paul Atreides"
mvtmj = mvtmj


def cheatsheet() -> str:
    return "mvtmj({}) -> Majority rule median voter."
