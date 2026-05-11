"""Bayesian spatial vote probability.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svbpl(data=None, n=50):
    """Bayesian spatial vote probability.

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


short = "svbpl"
alias = "svbpl"
quote = "The spice must flow. -- Paul Atreides"
svbpl = svbpl


def cheatsheet() -> str:
    return "svbpl({}) -> Bayesian spatial vote probability."
