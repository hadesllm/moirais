"""Cosine similarity voting model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svcos(data=None, n=50):
    """Cosine similarity voting model.

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


short = "svcos"
alias = "svcos"
quote = "In the midst of chaos, there is also opportunity. -- Sun Tzu"
svcos = svcos


def cheatsheet() -> str:
    return "svcos({}) -> Cosine similarity voting model."
