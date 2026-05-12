# morie.fn -- function file (hadesllm/morie)
"""
CO2 concentration spatial

Category: GeoClim
"""

import numpy as np


def gcco2(data=None, coords=None, n=50):
    """CO2 concentration spatial

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


short = "gcco2"
alias = "gcco2"
quote = "Get in the robot, Shinji! -- Misato"
gcco2 = gcco2


def cheatsheet() -> str:
    return "gcco2({}) -> CO2 concentration spatial"
