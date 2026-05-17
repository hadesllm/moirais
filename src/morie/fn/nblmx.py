# morie.fn -- function file (hadesllm/morie)
"""
Lmax peak level

Category: NoisBrd
"""

import numpy as np


def nblmx(data=None, coords=None, n=50):
    """Lmax peak level

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


short = "nblmx"
alias = "nblmx"
quote = "It is not what happens to you, but how you react, that matters. -- Epictetus"
nblmx = nblmx


def cheatsheet() -> str:
    return "nblmx({}) -> Lmax peak level"
