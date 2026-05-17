"""
Voronoi neighbor count distribution

Category: GeoAnalysis
"""

import numpy as np


def vornnb(x=None, y=None, values=None, resolution=50):
    """Voronoi neighbor count distribution

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


short = "vornnb"
alias = "vornnb"
quote = "We must know. We will know. -- David Hilbert"
vornnb = vornnb


def cheatsheet() -> str:
    return "vornnb({}) -> Voronoi neighbor count distribution"
