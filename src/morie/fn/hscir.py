# morie.fn — function file (hadesllm/morie)
"""Hotelling circular competition.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hscir(data=None, n=50):
    """Hotelling circular competition.

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


short = "hscir"
alias = "hscir"
quote = "The spice must flow. -- Paul Atreides"
hscir = hscir


def cheatsheet() -> str:
    return "hscir({}) -> Hotelling circular competition."
