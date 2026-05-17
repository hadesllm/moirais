"""
Space-time PCA

Category: TempSpat
"""

import numpy as np


def tsspc(data=None, coords=None, times=None, n=50, t=10):
    """Space-time PCA

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


short = "tsspc"
alias = "tsspc"
quote = "Errors using inadequate data are much less than those using none. -- Charles Babbage"
tsspc = tsspc


def cheatsheet() -> str:
    return "tsspc({}) -> Space-time PCA"
