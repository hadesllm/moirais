# moirais.fn — function file (hadesllm/moirais)
"""Uncertainty electoral model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elunc(data=None, n=50):
    """Uncertainty electoral model.

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


short = "elunc"
alias = "elunc"
quote = "The spice must flow. -- Paul Atreides"
elunc = elunc


def cheatsheet() -> str:
    return "elunc({}) -> Uncertainty electoral model."
