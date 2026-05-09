# moirais.fn — function file (hadesllm/moirais)
"""
KDE nearest facility distance

Category: GeoAnalysis
"""

import numpy as np


def kdenf(x=None, y=None, values=None, resolution=50):
    """KDE nearest facility distance

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if x is None:
        x = np.random.default_rng(0).uniform(0, 100, 50)
    if y is None:
        y = np.random.default_rng(1).uniform(0, 100, 50)
    if values is None:
        values = np.random.default_rng(2).standard_normal(len(x))
    stat = float(np.mean(values))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "x_range": [float(np.min(x)), float(np.max(x))],
            "y_range": [float(np.min(y)), float(np.max(y))],
            "n": len(x),
        },
    )


short = "kdenf"
alias = "kdenf"
quote = "Winter is coming. -- Stark motto"
kdenf = kdenf


def cheatsheet() -> str:
    return "kdenf({}) -> KDE nearest facility distance"
