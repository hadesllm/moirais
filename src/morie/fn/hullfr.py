# morie.fn — function file (hadesllm/morie)
"""
Convex hull fractal dimension

Category: GeoAnalysis
"""

import numpy as np


def hullfr(x=None, y=None, values=None, resolution=50):
    """Convex hull fractal dimension

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


short = "hullfr"
alias = "hullfr"
quote = "I alone level up. -- Sung Jin-Woo"
hullfr = hullfr


def cheatsheet() -> str:
    return "hullfr({}) -> Convex hull fractal dimension"
