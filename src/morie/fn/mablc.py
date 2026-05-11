# morie.fn — function file (hadesllm/morie)
"""
Bleaching risk spatial

Category: MarinSp
"""

import numpy as np


def mablc(depth=None, temp=None, salinity=None, coords=None, n=50):
    """Bleaching risk spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if depth is None:
        depth = np.random.default_rng(0).uniform(0, 5000, n)
    if temp is None:
        temp = np.random.default_rng(1).uniform(-2, 30, n)
    if salinity is None:
        salinity = np.random.default_rng(2).uniform(30, 40, n)
    if coords is None:
        coords = np.random.default_rng(3).uniform(-180, 180, (n, 2))
    stat = float(np.mean(temp))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "n": len(depth),
            "mean_depth": float(np.mean(depth)),
            "mean_temp": float(np.mean(temp)),
            "mean_salinity": float(np.mean(salinity)),
        },
    )


short = "mablc"
alias = "mablc"
quote = "You should enjoy the detours. -- Ging"
mablc = mablc


def cheatsheet() -> str:
    return "mablc({}) -> Bleaching risk spatial"
