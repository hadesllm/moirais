# morie.fn -- function file (hadesllm/morie)
"""
CH4 concentration spatial

Category: GeoClim
"""

import numpy as np


def gcch4(data=None, coords=None, n=50):
    """CH4 concentration spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "gcch4"
alias = "gcch4"
quote = "What is now proved was once only imagined. -- William Blake"
gcch4 = gcch4


def cheatsheet() -> str:
    return "gcch4({}) -> CH4 concentration spatial"
