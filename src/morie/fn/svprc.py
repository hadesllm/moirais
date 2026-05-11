"""Pure proximity for categorical issues.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svprc(data=None, n=50):
    """Pure proximity for categorical issues.

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


short = "svprc"
alias = "svprc"
quote = "The spice must flow. -- Paul Atreides"
svprc = svprc


def cheatsheet() -> str:
    return "svprc({}) -> Pure proximity for categorical issues."
