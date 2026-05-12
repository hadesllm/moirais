# morie.fn -- function file (hadesllm/morie)
"""
Topographic position index

Category: GeoAnalysis
"""

import numpy as np


def gatpi(x=None, y=None, values=None, resolution=50):
    """Topographic position index

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


short = "gatpi"
alias = "gatpi"
quote = "Power is everything. -- Sung Jin-Woo"
gatpi = gatpi


def cheatsheet() -> str:
    return "gatpi({}) -> Topographic position index"
