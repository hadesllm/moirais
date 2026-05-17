"""Saddle-point spatial utility.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def umsad(data=None, n=50):
    """Saddle-point spatial utility.

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


short = "umsad"
alias = "umsad"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
umsad = umsad


def cheatsheet() -> str:
    return "umsad({}) -> Saddle-point spatial utility."
