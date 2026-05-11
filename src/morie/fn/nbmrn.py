# morie.fn — function file (hadesllm/morie)
"""
Marine noise spatial

Category: NoisBrd
"""

import numpy as np


def nbmrn(data=None, coords=None, n=50):
    """Marine noise spatial

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


short = "nbmrn"
alias = "nbmrn"
quote = "I'm gonna be King of the Pirates! -- Luffy"
nbmrn = nbmrn


def cheatsheet() -> str:
    return "nbmrn({}) -> Marine noise spatial"
