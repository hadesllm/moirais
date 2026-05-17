# morie.fn -- function file (hadesllm/morie)
"""
Noise compliance spatial

Category: NoisBrd
"""

import numpy as np


def nbcmp(data=None, coords=None, n=50):
    """Noise compliance spatial

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


short = "nbcmp"
alias = "nbcmp"
quote = "What is now proved was once only imagined. -- William Blake"
nbcmp = nbcmp


def cheatsheet() -> str:
    return "nbcmp({}) -> Noise compliance spatial"
