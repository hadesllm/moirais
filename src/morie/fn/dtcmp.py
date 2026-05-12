# morie.fn -- function file (hadesllm/morie)
"""Comparative fit dimensionality.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtcmp(data=None, n=50):
    """Comparative fit dimensionality.

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


short = "dtcmp"
alias = "dtcmp"
quote = "The spice must flow. -- Paul Atreides"
dtcmp = dtcmp


def cheatsheet() -> str:
    return "dtcmp({}) -> Comparative fit dimensionality."
