# moirais.fn — function file (hadesllm/moirais)
"""
Underwater noise mapping

Category: NoisBrd
"""

import numpy as np


def nbund(data=None, coords=None, n=50):
    """Underwater noise mapping

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(30, 90, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean_db": float(np.mean(data)), "max_db": float(np.max(data))},
    )


short = "nbund"
alias = "nbund"
quote = "It's over 9000! -- Vegeta"
nbund = nbund


def cheatsheet() -> str:
    return "nbund({}) -> Underwater noise mapping"
