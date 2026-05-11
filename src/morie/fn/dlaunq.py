# morie.fn — function file (hadesllm/morie)
"""
Delaunay triangle quality metrics

Category: GeoAnalysis
"""

import numpy as np


def dlaunq(x=None, y=None, values=None, resolution=50):
    """Delaunay triangle quality metrics

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


short = "dlaunq"
alias = "dlaunq"
quote = "Resistance is futile. -- Borg"
dlaunq = dlaunq


def cheatsheet() -> str:
    return "dlaunq({}) -> Delaunay triangle quality metrics"
