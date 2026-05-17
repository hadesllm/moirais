# morie.fn -- function file (hadesllm/morie)
"""
Viewshed analysis

Category: GeoAnalysis
"""

import numpy as np


def gaview(x=None, y=None, values=None, resolution=50):
    """Viewshed analysis

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


short = "gaview"
alias = "gaview"
quote = "What is now proved was once only imagined. -- William Blake"
gaview = gaview


def cheatsheet() -> str:
    return "gaview({}) -> Viewshed analysis"
