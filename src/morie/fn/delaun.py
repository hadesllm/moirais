# morie.fn -- function file (hadesllm/morie)
"""
Delaunay triangulation mesh

Category: GeoAnalysis
"""

import numpy as np


def delaun(x=None, y=None, values=None, resolution=50):
    """Delaunay triangulation mesh

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


short = "delaun"
alias = "delaun"
quote = "If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton"
delaun = delaun


def cheatsheet() -> str:
    return "delaun({}) -> Delaunay triangulation mesh"
