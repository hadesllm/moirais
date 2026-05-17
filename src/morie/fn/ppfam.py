# morie.fn -- function file (hadesllm/morie)
"""Party family positioning.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppfam(data=None, n=50):
    """Party family positioning.

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


short = "ppfam"
alias = "ppfam"
quote = "We must know. We will know. -- David Hilbert"
ppfam = ppfam


def cheatsheet() -> str:
    return "ppfam({}) -> Party family positioning."
