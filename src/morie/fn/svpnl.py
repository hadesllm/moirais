"""Panel probit spatial vote.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svpnl(data=None, n=50):
    """Panel probit spatial vote.

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


short = "svpnl"
alias = "svpnl"
quote = "Logic is the foundation of all certain knowledge. -- Leonhard Euler"
svpnl = svpnl


def cheatsheet() -> str:
    return "svpnl({}) -> Panel probit spatial vote."
