# moirais.fn — function file (hadesllm/moirais)
"""
Ocean deoxygenation

Category: GeoClim
"""

import numpy as np


def gcdox(data=None, coords=None, n=50):
    """Ocean deoxygenation

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


short = "gcdox"
alias = "gcdox"
quote = "Fear is the mind-killer. -- Bene Gesserit"
gcdox = gcdox


def cheatsheet() -> str:
    return "gcdox({}) -> Ocean deoxygenation"
