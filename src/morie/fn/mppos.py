# morie.fn — function file (hadesllm/morie)
"""Position-competition model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mppos(data=None, n=50):
    """Position-competition model.

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


short = "mppos"
alias = "mppos"
quote = "The spice must flow. -- Paul Atreides"
mppos = mppos


def cheatsheet() -> str:
    return "mppos({}) -> Position-competition model."
