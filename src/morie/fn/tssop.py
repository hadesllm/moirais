"""
Space-time OPTICS

Category: TempSpat
"""

import numpy as np


def tssop(data=None, coords=None, times=None, n=50, t=10):
    """Space-time OPTICS

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


short = "tssop"
alias = "tssop"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
tssop = tssop


def cheatsheet() -> str:
    return "tssop({}) -> Space-time OPTICS"
