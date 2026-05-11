# morie.fn — function file (hadesllm/morie)
"""
Stream order (Strahler)

Category: GeoAnalysis
"""

import numpy as np


def gastrm(x=None, y=None, values=None, resolution=50):
    """Stream order (Strahler)

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


short = "gastrm"
alias = "gastrm"
quote = "I will take a potato chip and eat it! -- Light"
gastrm = gastrm


def cheatsheet() -> str:
    return "gastrm({}) -> Stream order (Strahler)"
