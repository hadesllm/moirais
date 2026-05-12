# morie.fn -- function file (hadesllm/morie)
"""Salience index composite.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def isidx(data=None, n=50):
    """Salience index composite.

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


short = "isidx"
alias = "isidx"
quote = "The spice must flow. -- Paul Atreides"
isidx = isidx


def cheatsheet() -> str:
    return "isidx({}) -> Salience index composite."
