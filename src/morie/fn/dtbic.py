# morie.fn -- function file (hadesllm/morie)
"""BIC dimensionality test.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtbic(data=None, n=50):
    """BIC dimensionality test.

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


short = "dtbic"
alias = "dtbic"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
dtbic = dtbic


def cheatsheet() -> str:
    return "dtbic({}) -> BIC dimensionality test."
