"""Pure directional for categorical.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svdrc(data=None, n=50):
    """Pure directional for categorical.

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


short = "svdrc"
alias = "svdrc"
quote = "The spice must flow. -- Paul Atreides"
svdrc = svdrc


def cheatsheet() -> str:
    return "svdrc({}) -> Pure directional for categorical."
