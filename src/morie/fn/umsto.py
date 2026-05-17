"""Stochastic utility maximization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def umsto(data=None, n=50):
    """Stochastic utility maximization.

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


short = "umsto"
alias = "umsto"
quote = "It is not the strongest that survives, but the most adaptable. -- Charles Darwin"
umsto = umsto


def cheatsheet() -> str:
    return "umsto({}) -> Stochastic utility maximization."
