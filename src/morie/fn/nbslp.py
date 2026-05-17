# morie.fn -- function file (hadesllm/morie)
"""
Sleep disturbance spatial

Category: NoisBrd
"""

import numpy as np


def nbslp(data=None, coords=None, n=50):
    """Sleep disturbance spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(30, 90, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean_db": float(np.mean(data)), "max_db": float(np.max(data))},
    )


short = "nbslp"
alias = "nbslp"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
nbslp = nbslp


def cheatsheet() -> str:
    return "nbslp({}) -> Sleep disturbance spatial"
