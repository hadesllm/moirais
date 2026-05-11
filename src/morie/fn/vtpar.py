"""Parallel vote trading.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def vtpar(data=None, n=50):
    """Parallel vote trading.

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


short = "vtpar"
alias = "vtpar"
quote = "The spice must flow. -- Paul Atreides"
vtpar = vtpar


def cheatsheet() -> str:
    return "vtpar({}) -> Parallel vote trading."
