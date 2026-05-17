"""Gains from trade vote.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def vtgai(data=None, n=50):
    """Gains from trade vote.

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


short = "vtgai"
alias = "vtgai"
quote = "It is not the strongest that survives, but the most adaptable. -- Charles Darwin"
vtgai = vtgai


def cheatsheet() -> str:
    return "vtgai({}) -> Gains from trade vote."
