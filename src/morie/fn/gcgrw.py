# morie.fn -- function file (hadesllm/morie)
"""
Growing season change

Category: GeoClim
"""

import numpy as np


def gcgrw(data=None, coords=None, n=50):
    """Growing season change

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


short = "gcgrw"
alias = "gcgrw"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
gcgrw = gcgrw


def cheatsheet() -> str:
    return "gcgrw({}) -> Growing season change"
