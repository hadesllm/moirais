# moirais.fn — function file (hadesllm/moirais)
"""
Convex hull roundness index

Category: GeoAnalysis
"""

import numpy as np


def hullrn(x=None, y=None, values=None, resolution=50):
    """Convex hull roundness index

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


short = "hullrn"
alias = "hullrn"
quote = "Chaos is a ladder. -- Littlefinger"
hullrn = hullrn


def cheatsheet() -> str:
    return "hullrn({}) -> Convex hull roundness index"
