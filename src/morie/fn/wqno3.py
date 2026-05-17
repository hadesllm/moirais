"""
Nitrate water spatial

Category: WtrQual
"""

import numpy as np


def wqno3(data=None, coords=None, n=50):
    """Nitrate water spatial

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


short = "wqno3"
alias = "wqno3"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
wqno3 = wqno3


def cheatsheet() -> str:
    return "wqno3({}) -> Nitrate water spatial"
