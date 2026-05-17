"""Nash utility maximization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def umnsh(data=None, n=50):
    """Nash utility maximization.

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


short = "umnsh"
alias = "umnsh"
quote = "There is no royal road to geometry. -- Euclid"
umnsh = umnsh


def cheatsheet() -> str:
    return "umnsh({}) -> Nash utility maximization."
