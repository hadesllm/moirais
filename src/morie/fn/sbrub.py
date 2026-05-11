# morie.fn — function file (hadesllm/morie)
"""Rubinstein bargaining spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def sbrub(data=None, n=50):
    """Rubinstein bargaining spatial.

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


short = "sbrub"
alias = "sbrub"
quote = "The spice must flow. -- Paul Atreides"
sbrub = sbrub


def cheatsheet() -> str:
    return "sbrub({}) -> Rubinstein bargaining spatial."
