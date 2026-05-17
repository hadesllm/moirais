"""Pairwise vote trading.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def vtpai(data=None, n=50):
    """Pairwise vote trading.

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


short = "vtpai"
alias = "vtpai"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
vtpai = vtpai


def cheatsheet() -> str:
    return "vtpai({}) -> Pairwise vote trading."
