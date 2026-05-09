# moirais.fn — function file (hadesllm/moirais)
"""
ENSO pattern change

Category: GeoClim
"""

import numpy as np


def gcedn(data=None, coords=None, n=50):
    """ENSO pattern change

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


short = "gcedn"
alias = "gcedn"
quote = "Believe it! -- Naruto"
gcedn = gcedn


def cheatsheet() -> str:
    return "gcedn({}) -> ENSO pattern change"
