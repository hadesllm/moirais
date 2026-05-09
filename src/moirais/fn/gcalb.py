# moirais.fn — function file (hadesllm/moirais)
"""
Albedo feedback spatial

Category: GeoClim
"""

import numpy as np


def gcalb(data=None, coords=None, n=50):
    """Albedo feedback spatial

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


short = "gcalb"
alias = "gcalb"
quote = "El Psy Kongroo. -- Okabe"
gcalb = gcalb


def cheatsheet() -> str:
    return "gcalb({}) -> Albedo feedback spatial"
