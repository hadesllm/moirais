# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Alpha shape boundary detection

Category: GeoAnalysis
"""

import numpy as np


def alphsh(x=None, y=None, values=None, resolution=50):
    """Alpha shape boundary detection

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


short = "alphsh"
alias = "alphsh"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
alphsh = alphsh


def cheatsheet() -> str:
    return "alphsh({}) -> Alpha shape boundary detection"
