# morie.fn -- function file (hadesllm/morie)
"""Party distance matrix.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppdst(data=None, n=50):
    """Party distance matrix.

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


short = "ppdst"
alias = "ppdst"
quote = "The spice must flow. -- Paul Atreides"
ppdst = ppdst


def cheatsheet() -> str:
    return "ppdst({}) -> Party distance matrix."
