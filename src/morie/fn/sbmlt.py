# morie.fn -- function file (hadesllm/morie)
"""Multilateral bargaining spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def sbmlt(data=None, n=50):
    """Multilateral bargaining spatial.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "sbmlt"
alias = "sbmlt"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
sbmlt = sbmlt


def cheatsheet() -> str:
    return "sbmlt({}) -> Multilateral bargaining spatial."
