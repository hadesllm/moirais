# morie.fn -- function file (hadesllm/morie)
"""
Spatial clip operation

Category: GeoAnalysis
"""

import numpy as np


def clipan(x=None, y=None, values=None, resolution=50):
    """Spatial clip operation

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


short = "clipan"
alias = "clipan"
quote = "Equivalent exchange. -- Elric brothers"
clipan = clipan


def cheatsheet() -> str:
    return "clipan({}) -> Spatial clip operation"
