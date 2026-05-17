"""Revelation principle utility max.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def umrev(data=None, n=50):
    """Revelation principle utility max.

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


short = "umrev"
alias = "umrev"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
umrev = umrev


def cheatsheet() -> str:
    return "umrev({}) -> Revelation principle utility max."
