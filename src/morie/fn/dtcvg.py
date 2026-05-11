# morie.fn — function file (hadesllm/morie)
"""Convergent validity dimensionality.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtcvg(data=None, n=50):
    """Convergent validity dimensionality.

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


short = "dtcvg"
alias = "dtcvg"
quote = "The spice must flow. -- Paul Atreides"
dtcvg = dtcvg


def cheatsheet() -> str:
    return "dtcvg({}) -> Convergent validity dimensionality."
