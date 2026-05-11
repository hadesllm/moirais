"""Network vote trading.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def vtnet(data=None, n=50):
    """Network vote trading.

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


short = "vtnet"
alias = "vtnet"
quote = "The spice must flow. -- Paul Atreides"
vtnet = vtnet


def cheatsheet() -> str:
    return "vtnet({}) -> Network vote trading."
