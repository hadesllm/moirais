# moirais.fn — function file (hadesllm/moirais)
"""
N2O concentration spatial

Category: GeoClim
"""

import numpy as np


def gcn2o(data=None, coords=None, n=50):
    """N2O concentration spatial

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


short = "gcn2o"
alias = "gcn2o"
quote = "Fear is the mind-killer. -- Bene Gesserit"
gcn2o = gcn2o


def cheatsheet() -> str:
    return "gcn2o({}) -> N2O concentration spatial"
