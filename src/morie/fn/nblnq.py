# morie.fn -- function file (hadesllm/morie)
"""
Lnight quiet level

Category: NoisBrd
"""

import numpy as np


def nblnq(data=None, coords=None, n=50):
    """Lnight quiet level

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


short = "nblnq"
alias = "nblnq"
quote = "You have power over your mind, not outside events. -- Marcus Aurelius"
nblnq = nblnq


def cheatsheet() -> str:
    return "nblnq({}) -> Lnight quiet level"
