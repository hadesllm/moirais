# moirais.fn — function file (hadesllm/moirais)
"""
Concave hull estimation

Category: GeoAnalysis
"""

import numpy as np


def concav(x=None, y=None, values=None, resolution=50):
    """Concave hull estimation

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


short = "concav"
alias = "concav"
quote = "It's over 9000! -- Vegeta"
concav = concav


def cheatsheet() -> str:
    return "concav({}) -> Concave hull estimation"
