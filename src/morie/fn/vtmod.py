"""Modular vote trading.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def vtmod(data=None, n=50):
    """Modular vote trading.

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


short = "vtmod"
alias = "vtmod"
quote = "We must know. We will know. -- David Hilbert"
vtmod = vtmod


def cheatsheet() -> str:
    return "vtmod({}) -> Modular vote trading."
