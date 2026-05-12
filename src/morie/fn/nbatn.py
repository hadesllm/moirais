# morie.fn -- function file (hadesllm/morie)
"""
Attenuation distance

Category: NoisBrd
"""

import numpy as np


def nbatn(data=None, coords=None, n=50):
    """Attenuation distance

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


short = "nbatn"
alias = "nbatn"
quote = "Dedicate your hearts! -- Erwin"
nbatn = nbatn


def cheatsheet() -> str:
    return "nbatn({}) -> Attenuation distance"
