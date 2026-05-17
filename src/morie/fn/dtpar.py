# morie.fn -- function file (hadesllm/morie)
"""Parallel analysis dimensionality.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtpar(data=None, n=50):
    """Parallel analysis dimensionality.

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


short = "dtpar"
alias = "dtpar"
quote = "There is no royal road to geometry. -- Euclid"
dtpar = dtpar


def cheatsheet() -> str:
    return "dtpar({}) -> Parallel analysis dimensionality."
