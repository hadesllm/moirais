# morie.fn -- function file (hadesllm/morie)
"""Presence/absence roll call.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rcprs(data=None, n=50):
    """Presence/absence roll call.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(data)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "rcprs"
alias = "rcprs"
quote = "The Analytical Engine weaves algebraic patterns. -- Ada Lovelace"
rcprs = rcprs


def cheatsheet() -> str:
    return "rcprs({}) -> Presence/absence roll call."
