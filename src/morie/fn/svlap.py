"""Laplace distribution vote probability.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svlap(data=None, n=50):
    """Laplace distribution vote probability.

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


short = "svlap"
alias = "svlap"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
svlap = svlap


def cheatsheet() -> str:
    return "svlap({}) -> Laplace distribution vote probability."
