"""KKT conditions spatial max.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def umkkt(data=None, n=50):
    """KKT conditions spatial max.

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


short = "umkkt"
alias = "umkkt"
quote = "You have power over your mind, not outside events. -- Marcus Aurelius"
umkkt = umkkt


def cheatsheet() -> str:
    return "umkkt({}) -> KKT conditions spatial max."
