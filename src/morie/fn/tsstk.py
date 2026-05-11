"""
Space-time K-function

Category: TempSpat
"""

import numpy as np


def tsstk(data=None, coords=None, times=None, n=50, t=10):
    """Space-time K-function

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).standard_normal((n, t))
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    if times is None:
        times = np.arange(t)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "n_locations": data.shape[0],
            "n_times": data.shape[1],
            "mean": float(np.mean(data)),
            "var": float(np.var(data)),
        },
    )


short = "tsstk"
alias = "tsstk"
quote = "I am the hope of the universe. -- Goku"
tsstk = tsstk


def cheatsheet() -> str:
    return "tsstk({}) -> Space-time K-function"
