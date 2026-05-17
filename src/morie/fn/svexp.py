"""Exponential vote probability decay.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svexp(data=None, n=50):
    """Exponential vote probability decay.

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


short = "svexp"
alias = "svexp"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
svexp = svexp


def cheatsheet() -> str:
    return "svexp({}) -> Exponential vote probability decay."
