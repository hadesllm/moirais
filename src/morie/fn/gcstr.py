# morie.fn -- function file (hadesllm/morie)
"""
Storm intensity change

Category: GeoClim
"""

import numpy as np


def gcstr(data=None, coords=None, n=50):
    """Storm intensity change

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


short = "gcstr"
alias = "gcstr"
quote = "Set your heart ablaze! -- Rengoku"
gcstr = gcstr


def cheatsheet() -> str:
    return "gcstr({}) -> Storm intensity change"
