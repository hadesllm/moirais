"""Discounting model (Matthews 1979).

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svdcm(data=None, n=50):
    """Discounting model (Matthews 1979).

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


short = "svdcm"
alias = "svdcm"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
svdcm = svdcm


def cheatsheet() -> str:
    return "svdcm({}) -> Discounting model (Matthews 1979)."
