# morie.fn -- function file (hadesllm/morie)
"""Cutting plane roll call.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rccut(data=None, n=50):
    """Cutting plane roll call.

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


short = "rccut"
alias = "rccut"
quote = "If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton"
rccut = rccut


def cheatsheet() -> str:
    return "rccut({}) -> Cutting plane roll call."
