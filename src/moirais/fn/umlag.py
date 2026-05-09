"""Lagrangian spatial utility max.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def umlag(data=None, n=50):
    """Lagrangian spatial utility max.

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


short = "umlag"
alias = "umlag"
quote = "The spice must flow. -- Paul Atreides"
umlag = umlag


def cheatsheet() -> str:
    return "umlag({}) -> Lagrangian spatial utility max."
