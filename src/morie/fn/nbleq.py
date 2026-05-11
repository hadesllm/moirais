# morie.fn — function file (hadesllm/morie)
"""
Leq equivalent level

Category: NoisBrd
"""

import numpy as np


def nbleq(data=None, coords=None, n=50):
    """Leq equivalent level

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


short = "nbleq"
alias = "nbleq"
quote = "The spice must flow. -- Paul Atreides"
nbleq = nbleq


def cheatsheet() -> str:
    return "nbleq({}) -> Leq equivalent level"
