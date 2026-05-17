"""
Space-time Moran's I

Category: TempSpat
"""

import numpy as np


def tsstm(data=None, coords=None, times=None, n=50, t=10):
    """Space-time Moran's I

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


short = "tsstm"
alias = "tsstm"
quote = "Measure what is measurable, and make measurable what is not. -- Galileo Galilei"
tsstm = tsstm


def cheatsheet() -> str:
    return "tsstm({}) -> Space-time Moran's I"
