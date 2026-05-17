"""Angular proximity model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svang(data=None, n=50):
    """Angular proximity model.

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


short = "svang"
alias = "svang"
quote = "Mathematics is the queen of the sciences. -- Carl Friedrich Gauss"
svang = svang


def cheatsheet() -> str:
    return "svang({}) -> Angular proximity model."
