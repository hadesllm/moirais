"""Ordered logit spatial model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svord(data=None, n=50):
    """Ordered logit spatial model.

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


short = "svord"
alias = "svord"
quote = "The spice must flow. -- Paul Atreides"
svord = svord


def cheatsheet() -> str:
    return "svord({}) -> Ordered logit spatial model."
