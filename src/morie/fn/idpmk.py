# morie.fn -- function file (hadesllm/morie)
"""Markov chain ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpmk(data=None, n=50):
    """Markov chain ideal point.

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


short = "idpmk"
alias = "idpmk"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
idpmk = idpmk


def cheatsheet() -> str:
    return "idpmk({}) -> Markov chain ideal point."
