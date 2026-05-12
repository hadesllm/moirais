# morie.fn -- function file (hadesllm/morie)
"""
Coral bleaching trend

Category: GeoClim
"""

import numpy as np


def gccor(data=None, coords=None, n=50):
    """Coral bleaching trend

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "gccor"
alias = "gccor"
quote = "Get in the robot, Shinji! -- Misato"
gccor = gccor


def cheatsheet() -> str:
    return "gccor({}) -> Coral bleaching trend"
