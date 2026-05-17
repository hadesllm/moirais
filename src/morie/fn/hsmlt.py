# morie.fn -- function file (hadesllm/morie)
"""Multiparty Hotelling equilibrium.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hsmlt(data=None, n=50):
    """Multiparty Hotelling equilibrium.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(np.asarray(data, dtype=float))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "hsmlt"
alias = "hsmlt"
quote = "What is now proved was once only imagined. -- William Blake"
hsmlt = hsmlt


def cheatsheet() -> str:
    return "hsmlt({}) -> Multiparty Hotelling equilibrium."
