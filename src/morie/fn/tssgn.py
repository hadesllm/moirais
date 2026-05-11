"""
Space-time GNN proxy

Category: TempSpat
"""

import numpy as np


def tssgn(data=None, coords=None, times=None, n=50, t=10):
    """Space-time GNN proxy

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


short = "tssgn"
alias = "tssgn"
quote = "Even the smallest person can change the future. -- Galadriel"
tssgn = tssgn


def cheatsheet() -> str:
    return "tssgn({}) -> Space-time GNN proxy"
