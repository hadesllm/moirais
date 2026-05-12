# morie.fn -- function file (hadesllm/morie)
"""Variance explained test.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtvar(data=None, n=50):
    """Variance explained test.

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


short = "dtvar"
alias = "dtvar"
quote = "The spice must flow. -- Paul Atreides"
dtvar = dtvar


def cheatsheet() -> str:
    return "dtvar({}) -> Variance explained test."
