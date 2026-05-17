"""Constrained vote trading.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def vtcon(data=None, n=50):
    """Constrained vote trading.

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


short = "vtcon"
alias = "vtcon"
quote = "The heart has its reasons of which reason knows nothing. -- Blaise Pascal"
vtcon = vtcon


def cheatsheet() -> str:
    return "vtcon({}) -> Constrained vote trading."
