# morie.fn — function file (hadesllm/morie)
"""Government formation spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpgov(data=None, n=50):
    """Government formation spatial.

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


short = "mpgov"
alias = "mpgov"
quote = "The spice must flow. -- Paul Atreides"
mpgov = mpgov


def cheatsheet() -> str:
    return "mpgov({}) -> Government formation spatial."
