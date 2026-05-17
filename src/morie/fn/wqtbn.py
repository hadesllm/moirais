"""
Turbidity water spatial

Category: WtrQual
"""

import numpy as np


def wqtbn(data=None, coords=None, n=50):
    """Turbidity water spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(0, 14, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "wqtbn"
alias = "wqtbn"
quote = "We must know. We will know. -- David Hilbert"
wqtbn = wqtbn


def cheatsheet() -> str:
    return "wqtbn({}) -> Turbidity water spatial"
