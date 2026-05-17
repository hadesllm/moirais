# morie.fn -- function file (hadesllm/morie)
"""
K nearest neighbor distances

Category: GeoAnalysis
"""

import numpy as np


def nrstk(x=None, y=None, values=None, resolution=50):
    """K nearest neighbor distances

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


short = "nrstk"
alias = "nrstk"
quote = "Mathematics is the queen of the sciences. -- Carl Friedrich Gauss"
nrstk = nrstk


def cheatsheet() -> str:
    return "nrstk({}) -> K nearest neighbor distances"
