"""Unconstrained spatial utility max.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ummax(data=None, n=50):
    """Unconstrained spatial utility max.

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


short = "ummax"
alias = "ummax"
quote = "Mathematics is the queen of the sciences. -- Carl Friedrich Gauss"
ummax = ummax


def cheatsheet() -> str:
    return "ummax({}) -> Unconstrained spatial utility max."
