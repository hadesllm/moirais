# moirais.fn — function file (hadesllm/moirais)
"""Rotation test dimensionality.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtrot(data=None, n=50):
    """Rotation test dimensionality.

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


short = "dtrot"
alias = "dtrot"
quote = "The spice must flow. -- Paul Atreides"
dtrot = dtrot


def cheatsheet() -> str:
    return "dtrot({}) -> Rotation test dimensionality."
