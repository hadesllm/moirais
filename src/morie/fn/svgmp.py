"""Gompertz spatial vote probability.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svgmp(data=None, n=50):
    """Gompertz spatial vote probability.

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


short = "svgmp"
alias = "svgmp"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
svgmp = svgmp


def cheatsheet() -> str:
    return "svgmp({}) -> Gompertz spatial vote probability."
