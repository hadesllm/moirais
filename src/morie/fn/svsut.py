"""Separable multidimensional utility.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svsut(data=None, n=50):
    """Separable multidimensional utility.

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


short = "svsut"
alias = "svsut"
quote = "In the midst of chaos, there is also opportunity. -- Sun Tzu"
svsut = svsut


def cheatsheet() -> str:
    return "svsut({}) -> Separable multidimensional utility."
