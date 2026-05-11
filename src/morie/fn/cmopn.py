# morie.fn — function file (hadesllm/morie)
"""Open rule committee.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def cmopn(data=None, n=50):
    """Open rule committee.

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


short = "cmopn"
alias = "cmopn"
quote = "The spice must flow. -- Paul Atreides"
cmopn = cmopn


def cheatsheet() -> str:
    return "cmopn({}) -> Open rule committee."
