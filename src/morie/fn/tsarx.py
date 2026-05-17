"""
Spatial ARX model

Category: TempSpat
"""

import numpy as np


def tsarx(data=None, coords=None, times=None, n=50, t=10):
    """Spatial ARX model

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


short = "tsarx"
alias = "tsarx"
quote = "The heart has its reasons of which reason knows nothing. -- Blaise Pascal"
tsarx = tsarx


def cheatsheet() -> str:
    return "tsarx({}) -> Spatial ARX model"
