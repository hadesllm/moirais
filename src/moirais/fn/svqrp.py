"""Quantile regression spatial prob.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svqrp(data=None, n=50):
    """Quantile regression spatial prob.

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


short = "svqrp"
alias = "svqrp"
quote = "The spice must flow. -- Paul Atreides"
svqrp = svqrp


def cheatsheet() -> str:
    return "svqrp({}) -> Quantile regression spatial prob."
