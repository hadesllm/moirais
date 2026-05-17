# morie.fn -- function file (hadesllm/morie)
"""Kurtosis-based polarization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plkur(data=None, n=50):
    """Kurtosis-based polarization.

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


short = "plkur"
alias = "plkur"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
plkur = plkur


def cheatsheet() -> str:
    return "plkur({}) -> Kurtosis-based polarization."
