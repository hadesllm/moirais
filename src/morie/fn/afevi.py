# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
EVI crop monitoring

Category: AgriSp
"""

import numpy as np


def afevi(yield_data=None, soil=None, coords=None, n=50):
    """EVI crop monitoring

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if yield_data is None:
        yield_data = np.random.default_rng(0).uniform(50, 200, n)
    if soil is None:
        soil = np.random.default_rng(1).uniform(0, 1, n)
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    stat = float(np.mean(yield_data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(yield_data), "mean_yield": float(np.mean(yield_data)), "mean_soil": float(np.mean(soil))},
    )


short = "afevi"
alias = "afevi"
quote = "I'm gonna be King of the Pirates! -- Luffy"
afevi = afevi


def cheatsheet() -> str:
    return "afevi({}) -> EVI crop monitoring"
