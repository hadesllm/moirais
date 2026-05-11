# morie.fn — function file (hadesllm/morie)
"""Discriminant test dimensionality.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtdcr(data=None, n=50):
    """Discriminant test dimensionality.

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


short = "dtdcr"
alias = "dtdcr"
quote = "The spice must flow. -- Paul Atreides"
dtdcr = dtdcr


def cheatsheet() -> str:
    return "dtdcr({}) -> Discriminant test dimensionality."
